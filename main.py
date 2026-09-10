
"""Prototype runner for proposed shared/specific mobility models."""
import argparse, torch
import importlib
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["model1","model2","model3","model4","model5"],default="model1")
    p.add_argument("--n-poi",type=int,default=5000)
    args=p.parse_args()
    m=importlib.import_module("models."+args.model)
    cls=getattr(m, args.model.title().replace("_",""))
    net=cls(args.n_poi)
    x=torch.randint(0,args.n_poi,(4,20))
    print(net(x).shape)
if __name__=="__main__": main()
