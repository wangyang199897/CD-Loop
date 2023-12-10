def run(chromosome_list, output_directory):
    def get_submatrix_netative(matrix_input_file_path, center_point_input_file_path, output_file_path,
                               negative_name_sort, chromosome):
        all_matrix_file = open(matrix_input_file_path, 'r')
        number_all_matrix_file = 0
        for line in all_matrix_file:
            number_all_matrix_file += 1
        all_matrix_file.close()

        center_point_file = open(center_point_input_file_path, 'r')
        point_list = []

        for line_center_point_file in center_point_file:
            temp = line_center_point_file.strip('\n').split('	')
            point_list.append([int(temp[0]), int(temp[1])])
        point_list_temp = sorted(point_list)

        num_parts = 100
        part_length = len(point_list_temp) // num_parts
        point_list_all = [[] for _ in range(num_parts)]

        for i in range(num_parts):
            start_index = i * part_length
            end_index = (i + 1) * part_length
            point_list_all[i] = point_list_temp[start_index:end_index]

        if len(point_list_temp) % num_parts != 0:
            point_list_all[-1].extend(point_list_temp[num_parts * part_length:])

        new_number_point_all = 0
        for point_list_part in range(100):
            point_list = point_list_all[point_list_part]
            number_point = len(point_list)

            all_matrix = []
            for i in range(number_point):
                submatrix = [[0.0] * 28 for _ in range(28)]
                all_matrix.append(submatrix)

            current_row = 0
            all_matrix_file = open(matrix_input_file_path, 'r')
            for line_all_matrix_file in all_matrix_file:
                current_row += 1
                line = line_all_matrix_file.split('\t')
                for num in range(number_point):
                    rows_point = point_list[num][0]
                    columns_point = point_list[num][1]
                    start_row = rows_point - 13
                    end_row = rows_point + 14
                    start_column = columns_point - 13
                    end_column = columns_point + 14
                    if start_row <= current_row <= end_row:
                        current_column = start_column
                        if start_column < 1:
                            current_column = 1  # 列从1开始
                        if end_column > len(line):
                            end_column = len(line)
                        while current_column <= end_column:
                            all_matrix[num][current_row - start_row][current_column - start_column] = line[
                                current_column - 1]
                            current_column += 1
                    elif current_row < start_row:
                        break

            all_matrix_file.close()
            center_point_file.close()

            point_list_new = point_list
            all_matrix_new = all_matrix

            del all_matrix
            del point_list
            new_number_point = len(point_list_new)
            new_number_point_all += new_number_point
            output_file_point_list = open(negative_name_sort[:-4] + '_' + str(point_list_part) + '.txt', 'w+')
            for num in point_list_new:
                output_file_point_list.write('chr' + chromosome + '\t' + str(num[0]) + '\t' + str(num[1]) + '\n')
            output_file_point_list.close()

            all_matrix_new = np.array(all_matrix_new)
            all_matrix_new = all_matrix_new.astype('float32')
            all_matrix_new = all_matrix_new.reshape(-1, 784)
            label_1 = np.zeros((new_number_point, 1))
            label_1 = label_1.astype('float32')
            all_all = np.concatenate((all_matrix_new, label_1), axis=1)
            np.save(output_file_path[:-4] + "_" + str(point_list_part) + '.npy', all_all)



    a = chromosome_list
    for n in range(len(a)):
        negative_name = output_directory + '/deleteInteractionPointInformation/center_point_remove_less1_chr' + str(a[n]) + '.txt'
        big_matrix_name = output_directory + '/bigmatrix/KR_matrix_5kb.chr' + str(a[n])
        np_save_delete_negative = output_directory + '/subMatrix/npy/KR_5kb_matrix_chr' + str(a[n]) + '_0_sort.npy'
        negative_center_delete_sort = output_directory + '/subMatrix/cp/negative_chr' + str(a[n]) + '-5KB-sort.txt'

        get_submatrix_netative(big_matrix_name, negative_name, np_save_delete_negative, negative_center_delete_sort, str(a[n]))

    for n in range(len(a)):
        data_list = []
        save_npy_path = output_directory + '/subMatrix/npy/KR_5kb_matrix_chr' + str(a[n]) + '_0_all.npy'
        for m in range(100):
            npy_path = output_directory + '/subMatrix/npy/KR_5kb_matrix_chr' + str(a[n]) + '_0_sort_' + str(m) + '.npy'
            array = np.load(npy_path)
            data_list.append(array)
        merged_array = np.concatenate(data_list, axis=0)
        np.save(save_npy_path, merged_array)
