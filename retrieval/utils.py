 
def recursive_print_dict( d, indent = 0 ):
    if not isinstance(d, list) and not isinstance(d, dict):
        print("\t" * (indent+1), d)
        return
        
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        elif isinstance(v, list):
            print("\t" * indent, f"{k}")
            allsingle = True
            for item in v:
                if isinstance(item, list) or isinstance(item, dict):
                    allsingle = False
            if allsingle:
                print("\t" * indent, v)
            else:
                for item in v:
                    recursive_print_dict(item, indent)
        else:
            print("\t" * indent, f"{k}:{v}")