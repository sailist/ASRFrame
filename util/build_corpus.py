
def build_py_corpus(fs,out_path):
    size = len(fs)
    with open(out_path,"w",encoding="utf-8") as w:
        for i,fn in enumerate(fs):
            with open(fn,encoding="utf-8") as f:
                f.readline()
                pyline = f.readline()
                w.write(f"{pyline.strip()}\n")

            print(f"\r{i}/{size}.",sep="\0",flush=True)

