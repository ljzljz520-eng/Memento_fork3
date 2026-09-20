def bg():
    from memento.background import Background

    backgound = Background()
    backgound.run()


def tl():
    from memento.timeline.timeline import Timeline

    t = Timeline()
    t.run()
