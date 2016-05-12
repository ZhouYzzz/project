class CheckError(Exception):
    pass

class CHECK():
    @classmethod
    def EQ(self,A,B):
        if (A == B):
            return
        else:
            raise CheckError(A,'Not Equal',B)

    @classmethod
    def NEQ(self,A,B):
        if (A == B):
            raise CheckError(A,'Equal',B)
        else:
            return

    @classmethod
    def GT(self,A,B):
        if (A >= B):
            return
        else:
            raise CheckError(A,'Not Greater than',B)

    @classmethod
    def LT(self,A,B):
        if (A <= B):
            return
        else:
            raise CheckError(A,'Not Less than',B)

    @classmethod
    def RD(self,A,B,alpha=1e-5):
        if ((A-B)>=alpha or (B-A)>=alpha):
            raise CheckError(A,'Not around',B)
        else:
            return