import code
import io
import sys


class RedirectSorties(io.TextIOBase):
    def __init__(self, sortieFic, sortie):
        self.F = sortieFic
        self.sortie = sortie
        #self.linebuf = ""
        
    def write(self, buf):
        if self.sortie:
            self.sortie.write(buf)
        #self.sortie.flush()
        print(buf, file=self.F, end="")
        #self.F.flush()
                
    def flush(self):
        self.sortie.flush()
        print("(flush)", file=self.F, end="")
        self.F.flush()
                
    

        

class Repl(code.InteractiveConsole):
    def __init__(self, nomFichier, *args, **kwargs):
        self.F = open(nomFichier, "w", encoding="utf-8")
        super().__init__(*args, **kwargs)
    
    def push(self, line):
        print("%s"%line, file=self.F)
        super().push(line)
    
    def raw_input(self, *args, **kwargs):
        #print("§§ input args=%s, kwargs=%s"%(repr(args), repr(kwargs)))
        return super().raw_input(*args, **kwargs)
        
    def interact(self, *args, **kwargs):
        sys.stdout, sys.stderr = RedirectSorties(self.F, sys.stdout), RedirectSorties(self.F, sys.stderr)
        super().interact(*args, **kwargs)


def main():
    repl = Repl("fichier_out.txt")
    repl.interact()
    
    
if __name__ == "__main__":
    main()