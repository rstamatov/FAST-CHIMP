start_t = 1050;
end_t = 1111;

for (i = 0; i < 5; i++) {
	run("Collect Garbage", "");
	wait(100);
}

selected = newArray();
deltaT = 1;
name_start = "D:/Rumen/FAST-CHIMP/results/propagated/t" + start_t + ".tif";
for (t = start_t; t <= end_t; t+= deltaT) {

	
	name = "D:/Rumen/FAST-CHIMP/results/propagated/t" + t + ".tif";
	if (t == 1000) { name = name_start; }

	run("Bio-Formats Importer", "open=[name] autoscale color_mode=Default rois_import=[ROI manager] view=[Standard ImageJ] stack_order=Default");

	if (t == start_t + deltaT) {
		run("Concatenate...", "open image1=[name_start] image2=[name]");
	} else if (t > start_t + deltaT) {
		run("Concatenate...", "open image1=[Untitled] image2=[name]");
	}

}
/*
for (label = 1; label < 200; label++) {
	if (selected.length > 0) {	
		if (label == selected[0]) {
			selected = Array.deleteIndex(selected, 0);
			run("Replace/Remove Label(s)", "label(s)=" + label + " final=" + 0);
		} else {
			//run("Replace/Remove Label(s)", "label(s)=" + label + " final=" + 200);
		} 
	} else {
		//run("Replace/Remove Label(s)", "label(s)=" + label + " final=" + 200);
	}
}


// due to a bug in the 3D viewer, it can't pick labels < 20.
// so we need to add 20 to each label
for (label = 200; label > 0; label--) {
	run("Replace/Remove Label(s)", "label(s)=" + label + " final=" + (label + 20));
}

run("Conversions...", " ");
setMinAndMax(0, 255);
*/
run("Stack to Hyperstack...", "order=xyczt(default) channels=1 slices=64 frames=" + ((end_t - start_t)/deltaT + 1) + " display=Composite");
//run("glasbey on dark");
//run("8-bit");
//setMinAndMax(0, 255);
//run("Properties...", "channels=1 slices=64 frames=" + ((end_t - start_t)/deltaT + 1) + " pixel_width=1.0000 pixel_height=1.0000 voxel_depth=2");
