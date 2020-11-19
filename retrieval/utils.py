 
def recursive_print_dict( d, indent = 0 ):
    if isinstance(d, str):
        print("\t" * (indent+1), d)
        return
    if isinstance(d, list):
        print("\t" * (indent+1), d)
        return
        
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        elif isinstance(v, list):
            print("\t" * indent, f"{k}")
            for item in v:
                recursive_print_dict(item, indent)
        else:
            print("\t" * indent, f"{k}:{v}")