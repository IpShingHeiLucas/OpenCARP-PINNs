# OpenCARP-PINNs

This cose is modified version of EP-PINNs developed by Clara Herrero Martin

This code contain sevarl input parameter
  **Reqauired Parameter**
      -m: The name of the output file
      -vf: The path to the vm.igb file
      -wf: The path to the V.igb file
      -ptf: The path to the .pts file. (BUT PLEASES DO NOT ADD .pts AT THE END)

  Please note that both Square1x_i and Square1x_e can be use interchangeably as OpenCARP-PINNs can only simulate monodomain

  
  **Optional Parameter**
    -w: to add W to the model input data
    -n: Add noise to the data
    -v: Solve the inverse problem, specify variables to predict (e.g. a / ad / abd'
    -ht: Predict heterogeneity - only in 2D
    -p: Plot voltage against time
    -a: Animation

**Common parameter example:**
  python main.py -m output_name -vf path/to/file/vm.igb -wf path/to/file/V.igb -p path/to/file/Square1x_i -p -a

    
    
