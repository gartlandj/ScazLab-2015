from PIL import Image
import scipy.io
import os
import glob

for folders in os.walk('/Users/jackgartland/Downloads/BSR_Cropped/BSDS500/data/images'):
	for folder in folders[1]:
		print folder
		os.chdir("/Users/jackgartland/Downloads/BSR_Cropped/BSDS500/data/images/" + folder)
		for filename in glob.glob('*.jpg'):
			img = Image.open(filename)
			img = img.crop((225, 145, 257, 177))
			img.save(filename)
		os.chdir("/Users/jackgartland/Downloads/BSR_Cropped/BSDS500/data/images")

i=0
for folders in os.walk('/Users/jackgartland/Downloads/BSR_Cropped/BSDS500/data/groundTruth'):
	for folder in folders[1]:
		print folder
		os.chdir("/Users/jackgartland/Downloads/BSR_Cropped/BSDS500/data/groundTruth/" + folder)
		for filename in glob.glob('*.mat'):
			mat_as_array = scipy.io.loadmat(filename)['groundTruth'][0][0][0][0][1]
			cropped_mat_as_array = mat_as_array[145:177, 225:257]
			if i==0:
				print cropped_mat_as_array
				print cropped_mat_as_array.shape
				i=1
			total_mat_file = scipy.io.loadmat(filename)
			total_mat_file['groundTruth'][0][0][0][0][0] = cropped_mat_as_array
			scipy.io.savemat(filename, total_mat_file)
		os.chdir("/Users/jackgartland/Downloads/BSR_Cropped/BSDS500/data/groundTruth")
