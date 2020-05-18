def global_format_func(s):
    fmt = {
        'country': "País",
        'state': 'Estado',
        'city': 'Município'
    }
    if s in fmt.keys():
        return fmt[s]
    else:
        return s
