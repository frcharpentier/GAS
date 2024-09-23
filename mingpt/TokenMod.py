from transformers import GPT2Tokenizer

class TokenMod(GPT2Tokenizer):
    def decodage(self, idx):
        return self.decoder[idx]
    
    def decode_token(self, token):
        etat = 0
        resu = ""
        interm = []
        #print(token)
        for cx in self.decoder[token]:
            x = self.byte_decoder[cx]
            if etat==0:
                if x > 127:
                    if 0xf0 <= x <= 0xf7:
                        etat = 3
                        interm.append(x)
                    elif 0xe0 <= x <= 0xef:
                        etat = 2
                        interm.append(x)
                    elif 0xc0 <= x <= 0xdf:
                        etat = 1
                        interm.append(x)
                    else:
                        resu += "\\x%02X"%x
                        interm = []
                else:
                    resu += chr(x)
                    etat = 0
            else:
                if 0x80 <= x <= 0xBF:
                    interm.append(x)
                    etat -= 1
                    if etat <= 0:
                        resu += bytes(interm).decode("utf-8")
                        interm = []
                else:
                    if len(interm) > 0:
                        for t in interm:
                            resu += "\\x%02X"%t
                        interm = []
                    resu += "\\x%02X"%x
                    etat = 0
        return resu
    
    def decode_visible(self, bpe_idx):
        resu = "·".join([self.decode_token(x) for x in bpe_idx])
        return resu