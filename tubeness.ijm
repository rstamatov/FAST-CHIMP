args = getArgument();
args_array = split(args, ",");
working_directory = args_array[1];

if (File.exists(working_directory + "/results/tubeness") == false) {
	File.makeDirectory(working_directory + "/results/tubeness");
}

filenames = getFileList(working_directory + "/results/restored");

for (t = 0; t < filenames.length; t++) {

	name = working_directory + "/results/restored/" + filenames[t];
	run("Bio-Formats Importer", "open=[" + name + "] autoscale color_mode=Default rois_import=[ROI manager] view=[Standard ImageJ] stack_order=Default");
	run("Properties...", "channels=1 slices=120 frames=1 pixel_width=0.0500000 pixel_height=0.0500000 voxel_depth=0.1800000");

	run("Tubeness", "sigma=0.28 use");
	
	saveAs("Tiff", working_directory + "/results/tubeness/" + filenames[t]);
	
	close();

	close();
}
