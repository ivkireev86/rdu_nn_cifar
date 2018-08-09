def auto_naming(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        name = str(type(self))
        for k, v in sorted(kwargs.items()):
            name += " {}={}".format(k, v)
        self.name = name

    return wrapper