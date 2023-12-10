def run(inf_file_path, chromosome_list, output_directory):
    def filter_and_save(file1_path, file2_path, output_path):
        interaction_data = {}

        with open(file1_path, "r") as f1:
            for line in f1:
                bin1, bin2, frequency = line.strip().split("\t")
                bin_start = int((int(bin1) / 5000) + 1)
                bin_end = int((int(bin2) / 5000) + 1)
                interaction_data[(bin_start, bin_end)] = float(frequency)
        with open(file2_path, "r") as f2, open(output_path, "w") as output:
            for line in f2:
                _, bin1, bin2 = line.strip().split("\t")
                bin1 = int(bin1)
                bin2 = int(bin2)
                if (bin1, bin2) in interaction_data and interaction_data[(bin1, bin2)] >= 1:
                    output.write(f"{bin1}\t{bin2}\n")

    def get_file_length(file_path):
        with open(file_path, "r") as file:
            line_count = sum(1 for _ in file)
        return line_count

    a = chromosome_list
    for i in range(len(a)):
        hic_read = inf_file_path + '/chr' + str(a[i]) + '-5kb.KRobserved'
        # 全部交互信息点，由fun2得到
        file_path_original = output_directory + '/interactionPointInformation/center_point_all_chr' + str(a[i]) + '.txt'
        # 删除交互频次小于1的所有点，输出删除后的中心点文件
        file_path_remove = output_directory + '/deleteInteractionPointInformation/center_point_remove_less1_chr' + str(a[i]) + '.txt'

        filter_and_save(hic_read, file_path_original, file_path_remove)
