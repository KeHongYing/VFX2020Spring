# LDR to HDR by Merging Multiple Images

&emsp; This is an algorithm to merge multiple images with different exposure time. We also use Robertson algorithm to find the response recovering curve and help us reconstruct the HDR image.

### Execution

&emsp; We use $python3$ to implement this algorithm.

```powershell
# install the required package
pip install -r requirement.txt

# Start to run the algorithm. Output is gR.npy, gG.npy, gB.npy, ER.npy, EG.npy, and EB.npy
python Robertson.py -d <img_directory_path> -i <iteration_time> [-s <shape>]

# Start to reconstruct Response Curve. Output is RecoverResponseCurve.png
python RecoverResponseCurve.py [-r <gR> -g <gG> -b <gB>]

# Start to reconstruct radiance map and HDR image. Output is RGB.png, radiance.png, and radiance.exr
python radiance_and_HDR.py [-r <ER> -g <EG> -b <EB>]
```

