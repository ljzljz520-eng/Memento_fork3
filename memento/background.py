import mss
import numpy as np
import cv2
import json
import datetime
import memento.utils as utils
import asyncio
import os
import time
import multiprocessing
from multiprocessing import Queue
import signal
from memento.OCR import Tesseract
from memento.caching import MetadataCache
from langchain.embeddings.openai import OpenAIEmbeddings
from memento.db import Db
from langchain.vectorstores import Chroma
from memento.segments import AppSegments
from memento.manifest import Manifest
from memento.purge import PurgeExecutor


class Background:
    def __init__(self):
        os.makedirs(utils.CACHE_PATH, exist_ok=True)
        existing = len(
            [f for f in os.listdir(utils.CACHE_PATH) if f.endswith(".mp4")]
        ) > 0 or os.path.exists(
            os.path.join(utils.CACHE_PATH, "manifest.json")
        )
        if existing:
            print("EXISTING MEMENTO CACHE FOUND")
            print("Continue this recording or erase and start over ? ")
            print("1. Continue")
            print("2. Erase and start over")
            choice = input("Choice: ")
            while choice not in ["1", "2"]:
                print("Please choose 1 or 2")
                choice = input("Choice: ")

            if choice == "2":
                os.system("rm -rf " + utils.CACHE_PATH)
                os.makedirs(utils.CACHE_PATH, exist_ok=True)

        # Loading the manifest migrates legacy dense files on first continue.
        self.manifest = Manifest.load(utils.CACHE_PATH)

        self.metadata_cache = MetadataCache(self.manifest)
        self.db = Db()

        # Finish any purge interrupted by a crash before anything else.
        self.chromadb = self._setup_chroma()
        PurgeExecutor(self.db, self.manifest, self.chromadb).resume()

        # A segment left unsealed by a previous crash cannot be appended to:
        # seal it and start a fresh segment.
        open_seg = self.manifest.open_segment()
        if open_seg is not None:
            self.manifest.seal(open_seg["uid"])
            self.manifest.save()
        self.segment = self.manifest.allocate_segment()
        self.manifest.save()

        self.sct = mss.mss()
        self.rec = utils.Recorder(
            self.manifest.abs_path(self.segment["media_path"])
        )
        self.rec.start()

        self.running = True
        self.frames_in_segment = 0

        self.app_segments = AppSegments()

        self.images_queue = Queue()
        self.results_queue = Queue()
        self.nb_workers = 1  # Need to have only one worker for segments for now. Seems to be ok performance wise
        self.workers = []
        self.pids = []
        for i in range(self.nb_workers):
            w = multiprocessing.Process(target=self.process_images, args=())
            self.workers.append(w)
            self.pids.append(w.pid)
        for i in range(self.nb_workers):
            self.workers[i].start()
            print("started worker", i)

    def _setup_chroma(self):
        if "OPENAI_API_KEY" not in os.environ:
            print(
                "No OPENAI_API_KEY environment variable found, LLM related features will not be available"
            )
            return None
        return Chroma(
            persist_directory=utils.CACHE_PATH,
            embedding_function=OpenAIEmbeddings(),
            collection_name="memento_db",
        )

    def process_images(self):
        # Infinite worker

        ocr = Tesseract()

        signal.signal(signal.SIGINT, self.stop_process)
        while True:
            start = time.time()
            data = self.images_queue.get()

            capture_id = data["capture_id"]
            im = data["im"]
            prev_im = data["prev_im"]
            window_title = data["window_title"]
            t = data["time"]
            diffscore = utils.imgdiff(im, prev_im)
            if diffscore < 0.1:  # TODO tune this
                results = []
                print("Skipping frame", capture_id, "because of imgdiff score ", diffscore)
            elif window_title == "memento-timeline":
                results = []
                print("Skipping frame", capture_id, "because looking at the timeline")
            else:
                start = time.time()
                results = ocr.process_image(im)
                print("Processing time :", time.time() - start)

            self.results_queue.put(
                {
                    "capture_id": capture_id,
                    "results": results,
                    "time": t,
                    "window_title": window_title,
                }
            )

    def stop_rec(self, sig, frame):
        # self.rec.stop()
        print("STOPPING MAIN", os.getpid())
        exit()

    def stop_process(self, sig, frame):
        print("STOPPING PROCESS", os.getpid())
        exit()

    def run(self):
        signal.signal(signal.SIGINT, self.stop_rec)

        print("Running in background ...")
        prev_im = np.zeros(
            (utils.RESOLUTION[1], utils.RESOLUTION[0], 3), dtype=np.uint8
        )
        while self.running:
            window_title = utils.get_active_window()

            # Get screenshot and add it to recorder
            im = np.array(self.sct.grab(self.sct.monitors[1]))
            im = im[:, :, :-1]
            im = cv2.resize(im, utils.RESOLUTION)
            asyncio.run(self.rec.new_im(im))

            # Stable identity + manifest registration come first, so the
            # capture is always resolvable by the time metadata is written.
            t = json.dumps(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            capture_id = self.manifest.allocate_capture_id()
            self.manifest.register_frame(
                self.segment, capture_id, window_title, t
            )
            self.manifest.save()
            self.frames_in_segment += 1

            self.images_queue.put(
                {
                    "im": im,
                    "prev_im": prev_im,
                    "window_title": window_title,
                    "time": t,
                    "capture_id": capture_id,
                }
            )
            prev_im = im

            self.metadata_cache.write(
                capture_id,
                {
                    "window_title": window_title,
                    "time": t,
                },
            )

            # deque results
            getting = True
            while getting:
                try:
                    result = self.results_queue.get(False)
                    bbs = []
                    text = []
                    all_text = ""
                    for i in range(len(result["results"])):
                        bb = {}
                        bb["x"] = result["results"][i]["x"]
                        bb["y"] = result["results"][i]["y"]
                        bb["w"] = result["results"][i]["w"]
                        bb["h"] = result["results"][i]["h"]
                        text.append(result["results"][i]["text"])
                        bbs.append(bb)
                        all_text += result["results"][i]["text"] + " "

                    result_capture_id = result["capture_id"]
                    frame_metadata = self.metadata_cache.get_frame_metadata(
                        result_capture_id
                    )
                    frame_metadata["bbs"] = bbs
                    frame_metadata["text"] = text
                    self.metadata_cache.write(result_capture_id, frame_metadata)
                    if len(text) == 0:
                        continue

                    self.app_segments.add(
                        result["window_title"],
                        result_capture_id,
                        all_text,
                        result["time"],
                    )
                    add_db_start = time.time()
                    try:
                        self.db.add_texts(
                            texts=text,
                            bbs=bbs,
                            frame_i=result_capture_id,
                            window_title=frame_metadata["window_title"],
                            time=frame_metadata["time"],
                        )
                        done_segments = self.app_segments.get_done_segments()
                        if len(done_segments) > 0:
                            print("DONE SEGMENTS", len(done_segments))
                            all_texts = []
                            for app_segment in done_segments:
                                for (
                                    content_segment
                                ) in app_segment.content_segments.segments:
                                    all_texts.append(content_segment.get_merged_text())

                            _id = str(list(done_segments[0].frames.keys())[0])
                            _time = done_segments[0].time
                            _window_title = done_segments[0].app_name
                            md = [
                                {
                                    "id": _id,
                                    "time": _time,
                                    "window_title": _window_title,
                                }
                                for _ in range(len(all_texts))
                            ]
                            if self.chromadb is not None:
                                self.chromadb.add_texts(
                                    texts=all_texts,
                                    metadatas=md,
                                )
                            print("ADDED SEQUENCE ", all_texts)
                        print("ADD TO DB TIME:", time.time() - add_db_start)
                    except Exception as e:
                        print("================aaaaaaa", e)
                        print("text", text)
                        print("===============")

                except Exception:
                    getting = False

            print("QUEUE SIZE", self.images_queue.qsize())

            if self.frames_in_segment >= utils.FPS * utils.SECONDS_PER_REC:
                print("CLOSE")
                self.rec.stop()
                self.manifest.seal(self.segment["uid"])
                self.manifest.save()

                self.segment = self.manifest.allocate_segment()
                self.manifest.save()
                self.frames_in_segment = 0
                self.rec = utils.Recorder(
                    self.manifest.abs_path(self.segment["media_path"])
                )
                self.rec.start()
