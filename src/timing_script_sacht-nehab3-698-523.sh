# Important note: this script must start with line 9 of filter.h being SAMPDIM 64
# Otherwise the conclusions are going to be meaningless.
# If you run the script until the end, it will finish with the correct setting for a next run
echo "8 SAMPLES" > timing-698-523.dat
sed -i -e 's/SAMPDIM 64/SAMPDIM 8/g' filter.h
make
echo "Thershold" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e threshold[0.35,0.57] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Unsharp" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e unsharp_mask[4.00,5.00,0.20] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Brightness" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e brightness_contrast[-0.30,0.80] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Emboss" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e emboss[5.00,1.00] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Laplacian" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplacian[] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Sharpening" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplace_edge_enhancement[1.50] --output test.png tucan-698-523.png >> timing-698-523.dat

echo "16 SAMPLES" >> timing-698-523.dat
sed -i -e 's/SAMPDIM 8/SAMPDIM 16/g' filter.h
make
echo "Thershold" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e threshold[0.35,0.57] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Unsharp" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e unsharp_mask[4.00,5.00,0.20] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Brightness" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e brightness_contrast[-0.30,0.80] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Emboss" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e emboss[5.00,1.00] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Laplacian" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplacian[] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Sharpening" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplace_edge_enhancement[1.50] --output test.png tucan-698-523.png >> timing-698-523.dat

echo "32 SAMPLES" >> timing-698-523.dat
sed -i -e 's/SAMPDIM 16/SAMPDIM 32/g' filter.h
make
echo "Thershold" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e threshold[0.35,0.57] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Unsharp" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e unsharp_mask[4.00,5.00,0.20] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Brightness" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e brightness_contrast[-0.30,0.80] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Emboss" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e emboss[5.00,1.00] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Laplacian" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplacian[] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Sharpening" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplace_edge_enhancement[1.50] --output test.png tucan-698-523.png >> timing-698-523.dat

echo "64 SAMPLES" >> timing-698-523.dat
sed -i -e 's/SAMPDIM 32/SAMPDIM 64/g' filter.h
make
echo "Thershold" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e threshold[0.35,0.57] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Unsharp" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e unsharp_mask[4.00,5.00,0.20] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Brightness" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e brightness_contrast[-0.30,0.80] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Emboss" >> timing-698-523.dat
./nlfilter --post sacht-nehab3 --pre card-bspline3 -e emboss[5.00,1.00] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Laplacian" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplacian[] --output test.png tucan-698-523.png >> timing-698-523.dat
echo "Sharpening" >> timing-698-523.dat
./nlfilter --post card-bspline3 --pre card-bspline3 -e laplace_edge_enhancement[1.50] --output test.png tucan-698-523.png >> timing-698-523.dat