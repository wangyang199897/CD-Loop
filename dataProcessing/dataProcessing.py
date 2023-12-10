import getBigMatrix
import getInpInfo
import delInpInfo
import getSubMatrix
import delSubMatrix
import getPicture


def run(inf_file_path, nof_file_path, chromosome_list, low_cp_scope, high_cp_scope, output_directory):
    getBigMatrix.run(inf_file_path, nof_file_path, chromosome_list, output_directory)
    getInpInfo.run(nof_file_path, chromosome_list, low_cp_scope, high_cp_scope, output_directory)
    delInpInfo.run(inf_file_path, chromosome_list, output_directory)
    getSubMatrix.run(chromosome_list, output_directory)
    delSubMatrix.run(chromosome_list, output_directory)
    getPicture.run(chromosome_list, output_directory)

