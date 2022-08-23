# ArchComp

This is a simple code for architecture basic block visualization. 

The optional architectures include : BiRealNet, ReActNet, BoolNetV1, BoolNetV2, BNext

Users can get the computation graph file using the ```tensorboard``` library, and the generated graph will be stored in the ```runs``` dir:
```py
  python ArchComparision.py --arc BNext --inplanes 64 --out_planes 64
```

After that, we can visualize the computing graph using the following command: 
```py
  tensorboard --logdir runs/ --bind_all
```

