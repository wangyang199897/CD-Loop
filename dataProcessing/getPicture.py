import numpy as np
import os
from PIL import Image


def run(chromosome_list, output_directory):
    def plot_hic_heatmap(hic_matrix, save_path):
        normalized_matrix = ((hic_matrix - np.min(hic_matrix)) / (np.max(hic_matrix) - np.min(hic_matrix))) * 255
        img = Image.fromarray(normalized_matrix.astype(np.uint8))
        resized_img = img.resize((28, 28), Image.ANTIALIAS)
        resized_img.save(save_path)

    a = chromosome_list
    for i in range(len(a)):
        infile1 = output_directory + '/deleteSubMatrix/KR_5kb_matrix_chr' + str(a[i]) + '_finall.npy'
        outfile1 = '../../predict/classification/data/test-images/0'

        folder_path = outfile1
        file_list = os.listdir(folder_path)
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        num_pos = 0
        arr1 = np.load(infile1)
        for m in range(len(arr1)):
            x = np.reshape(arr1[m][:-1], (28, 28))
            num_pos += 1
            save_path = outfile1 + '/' + 'chr' + str(a[i]) + '_' + str(num_pos) + '.png'
            plot_hic_heatmap(x, save_path)
