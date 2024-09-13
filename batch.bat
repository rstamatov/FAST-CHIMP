set working_directory="D:/Rumen/FAST-CHIMP/"
set start_t=1000
set end_t=1200

python TiffToStacks.py -filename "D:/Rumen/FISH chr 2/Experiment-484-Airyscan Processing-01.czi" -pixel_size 0.0425525 -pixel_size_z

python CARE_denoising.py

"C:/Users/marvi/Desktop/Fiji.app/ImageJ-win64.exe" -macro tubeness.ijm ,%working_directory%

python overseg_and_resize.py 0.01 0.1

python EMBEDSEG_predict.py

python refine_embedseg.py

python simple_elastix_register_pairs.py

python propagate.py %start_t% %end_t%



