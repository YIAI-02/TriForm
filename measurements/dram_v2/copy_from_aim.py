def store_to_DRAM_single_bank(self, dimm_index, channel_index, bank_index, row_index, col_index, size, data, op_trace):
        # GDDR6 stores with 32B granularity
        if op_trace and dimm_index == 0:
            for i in range((size - 1) // self.burst_length + 1):
                self.file.write("W MEM {} {} {}\n".format(channel_index, bank_index, row_index))
        self.pim_device["dimm_" + str(dimm_index)].dimm["channel_" + str(channel_index)].channel["bank_" + str(bank_index)].arrays[row_index][col_index : col_index + size] = data
    
def store_to_DRAM_all_banks(self, dim_iter, channel, row_current_head, seq, head, xv_data, num_rows_per_seq, rows_per_dim):
    for bank in range(self.num_banks):
        dim = dim_iter * self.num_banks + bank
        row_offset = num_rows_per_seq - 1
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + 1
        self.store_to_DRAM_single_bank(0, channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
    
def load_from_DRAM_single_bank(self, dimm_index, channel_index, bank_index, row_index, col_index, size, op_trace):
    if op_trace and dimm_index == 0:
        for i in range((size - 1) // self.burst_length + 1):
            self.file.write("R MEM {} {} {}\n".format(channel_index, bank_index, row_index))
    return self.pim_device["dimm_" + str(dimm_index)].dimm["channel_" + str(channel_index)].channel["bank_" + str(bank_index)].arrays[row_index][col_index : col_index + size]

def store_to_DRAM_multi_channel(self, data, row_index, mode, op_trace):
        if mode == self.mode["cache_k"]:
            # Store each seq in a bank
            seqlen = data.shape[0]
            shape = data.shape
            for seq in range(seqlen):
                dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
                data_seq = data[seq].reshape(-1)
                rows = self.head_dim * self.n_kv_heads // self.DRAM_column
                for row in range(rows):
                    data_row = data_seq[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + seq // self.FC_total_banks * rows + row, 0, self.DRAM_column, data_row, op_trace)  
                              
        elif mode == self.mode["cache_v"]:
            if self.intra_device_attention:
                seqlen = data.shape[-1]
                shape = data.shape
                rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iterations = self.head_dim // self.num_banks
                for channel in range(self.channels_per_block):
                    if channel == self.channels_per_block - 1:
                        num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                    else:
                        num_heads_iteration = num_heads_per_bank
                    for head_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains num_heads_per_bank heads
                        head = channel * num_heads_per_bank + head_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        row_current_head = row_index + (rows_per_dim * dim_iterations) * head_per_bank
                        for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                            for bank in range(self.num_banks):
                                dim = dim_iter * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
            else:
                seqlen = data.shape[-1]
                shape = data.shape
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                print("banks_per_head: ", banks_per_head)
                print("channels_per_head: ", channels_per_head)
                print("devices_per_head: ", devices_per_head)
                print("dim_iterations: ", dim_iterations)
                print("rows_per_seq_iteration: ", rows_per_seq_iteration)
                print("seq_iterations: ", seq_iterations)
                print("rows_per_seq: ", rows_per_seq)
                print("channels_per_row_offset: ", channels_per_row_offset)
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel%channels_per_head) * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            for row_offset in range(rows_per_seq):
                                if (channel % channels_per_head) // channels_per_row_offset == row_offset:
                                    if row_offset == rows_per_seq - 1:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, seqlen - self.DRAM_column * row_offset, data[head][dim][row_offset * self.DRAM_column:], op_trace)
                                    else:
                                        self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, self.DRAM_column, data[head][dim][row_offset * self.DRAM_column:(row_offset + 1) * self.DRAM_column], op_trace)
        else:
            bank_dim = (data.shape[0] - 1) // self.total_banks + 1
            utilized_banks = (data.shape[0] - 1) // bank_dim + 1
            # print(data.shape, bank_dim, utilized_banks)
            if mode == self.mode["weights"]:
                if self.model_parallel:
                    bank_dim = (data.shape[0] - 1) // self.FC_total_banks + 1
                    utilized_banks = (data.shape[0] - 1) // bank_dim + 1
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    vector_length = data.shape[1]
                    rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                    if i < utilized_banks - 1:
                        num_vectors = bank_dim
                        shape = data[i * num_vectors : (i+1) * num_vectors].shape
                    else:
                        num_vectors = data.shape[0] - bank_dim * (utilized_banks - 1)
                    for vector in range(num_vectors):
                        data_vector = data[i * bank_dim + vector]
                        for row in range(rows_per_vector):
                            # print(i * bank_dim + vector, dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row)
                            if row == rows_per_vector - 1:
                                data_tmp = data_vector[row * self.DRAM_column:]
                                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, vector_length - row * self.DRAM_column, data_tmp, op_trace)
                            else:
                                data_tmp = data_vector[row * self.DRAM_column:(row + 1) * self.DRAM_column]
                                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, self.DRAM_column, data_tmp, op_trace)
                # print(shape)
            elif mode == self.mode["vector"]:
                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim : (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim :]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif mode == self.mode["score"]:
                for i in range(self.total_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    data_bank = data[i].reshape(-1)
                    shape = data_bank.shape
                    data_size = data_bank.shape
                    rows = (data_size[0] - 1) // self.DRAM_column + 1
                    for row in range(rows-1):
                        data_tmp = data_bank[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row, 0, self.DRAM_column, data_tmp, op_trace)
                    data_tmp = data_bank[(rows-1) * self.DRAM_column:]
                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows - 1, 0, data_tmp.shape[0], data_tmp, op_trace)
            elif "vector_bank_group" in mode:
                # Gather the values in 4 banks in a bank group to 1 bank
                bank_dim = (data.shape[0] - 1) // (self.total_banks // 4) + 1
                utilized_banks = (data.shape[0] - 1) // bank_dim + 1

                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                neighbor_bank_index = int(mode[-1])
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank*4+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     bank_group_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*4+bank_group_index)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim: (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim:]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif "vector_neighbor_bank" in mode:
                # Gather the values in 2 neighboring banks in 1 bank
                bank_dim = (data.shape[0] - 1) // (self.total_banks // 2) + 1
                utilized_banks = (data.shape[0] - 1) // bank_dim + 1
                # print(data.shape, bank_dim, utilized_banks)
                shape=data[:bank_dim].shape
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                # e.g. Llama2-7B model has 4096 dim, with 10 channels, there are total 160 banks, bank_dim = 4096 // 80 + 1 = 52, bursts_per_bank = 52 // 16 + 1 = 4
                neighbor_bank_index = int(mode[-1])
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank*2+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            else:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: (bank + 1) * bank_dim]
                        else:
                            last_bank_length = data.shape[0] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length: bank * bank_dim + (burst + 1) * self.burst_length]
                            elif burst == last_bank_bursts - 1:
                                data_bank = data[bank * bank_dim + burst * self.burst_length:]
                            else:
                                continue
                        self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*2+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         data_bank = data[i * bank_dim: (i+1) * bank_dim]
                #         shape = data_bank.shape
                #     else:
                #         data_bank = data[i * bank_dim:]
                #     self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, data_bank.shape[0], data_bank, op_trace)
            elif "scores_bank_group" in mode:
                seqlen = data.shape[-1]
                # 4k scores use at most 4 rows
                # store each score in one bank, requring n_head banks
                rows_per_score = (seqlen - 1) // self.DRAM_column + 1
                num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
                shape = data.shape
                bank_group_index = int(mode[-1])
                for k in range(rows_per_score):
                    if k == rows_per_score - 1:
                        bank_dim = seqlen - self.DRAM_column * (rows_per_score - 1)
                    else:
                        bank_dim = self.DRAM_column
                    bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                    for j in range(num_heads_per_bank):
                        for burst in range(bursts_per_bank):
                            for bank in range(self.total_banks):
                                dimm_index, channel_index, bank_index = self.bank_index(bank)
                                if bank % 4 == bank_group_index:
                                    head = (bank // 4) * num_heads_per_bank + j
                                    if head > self.n_heads - 1:
                                        break
                                    if burst < bursts_per_bank - 1:
                                        data_bank = data[head][k * self.DRAM_column + burst * self.burst_length : k * self.DRAM_column + (burst + 1) * self.burst_length]
                                    else:
                                        data_bank = data[head][k * self.DRAM_column + burst * self.burst_length :]
                                    self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, burst * self.burst_length, data_bank.shape[0], data_bank, op_trace)

                # for i in range(self.total_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     bank_group_index = int(mode[-1])
                #     shape = data.shape
                #     if i % 4 == bank_group_index:
                #         for j in range(num_heads_per_bank):
                #             head = (i // 4) * num_heads_per_bank + j
                #             if head > self.n_heads - 1:
                #                 break
                #             for k in range(rows_per_score):
                #                 if k == rows_per_score - 1:
                #                     data_bank = data[head][k * self.DRAM_column :]
                #                 else:
                #                     data_bank = data[head][k * self.DRAM_column : (k + 1) * self.DRAM_column]
                #                 self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, data_bank.shape[0], data_bank, op_trace)
        return shape
    
    def load_from_DRAM_multi_channel(self, shape, row_index, mode, offset, op_trace):
        result = []
        if mode == self.mode["cache_k"]:
            seqlen = offset
            for seq in range(seqlen):
                seqs = []
                dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
                rows = self.head_dim * self.n_kv_heads // self.DRAM_column
                for row in range(rows):
                    seqs.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + seq // self.FC_total_banks * rows + row, 0, self.DRAM_column, op_trace))
                result.append(torch.cat(seqs).reshape(-1))
        elif mode == self.mode["cache_v"]:
            if self.intra_device_attention:
                seqlen = offset
                rows_per_seq = (seqlen - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                for channel in range(self.channels_per_block):
                    if channel == self.channels_per_block - 1:
                        num_heads_iteration = self.n_kv_heads - num_heads_per_bank * (self.channels_per_block - 1)
                    else:
                        num_heads_iteration = num_heads_per_bank
                    dim_iterations = self.head_dim // self.num_banks
                    for head_per_bank in range(num_heads_iteration):     # each head is distributed into all banks in a channel, each bank contains num_heads_per_bank heads
                        result_head = []
                        head = channel * num_heads_per_bank + head_per_bank
                        if head > self.n_kv_heads - 1:
                            break
                        row_current_head = row_index + (rows_per_dim * dim_iterations) * head_per_bank
                        for dim_iter in range(dim_iterations):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                            for bank in range(self.num_banks):
                                result_dim = []
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        result_dim.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_dim.append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 0, self.DRAM_column, op_trace))
                                result_head.append(torch.cat(result_dim))
                        result.append(torch.cat(result_head))
            else:
                seqlen = offset
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                result_heads = [[[] for dim in range(self.head_dim)] for head in range(self.n_kv_heads)]
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel%channels_per_head) * self.num_banks + bank
                                for row_offset in range(rows_per_seq):
                                    if row_offset == rows_per_seq - 1:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset * dim_iterations + dim_iter, 0, self.DRAM_column, op_trace))
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            for row_offset in range(rows_per_seq):
                                if (channel % channels_per_head) // channels_per_row_offset == row_offset:
                                    if row_offset == rows_per_seq - 1:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, seqlen - self.DRAM_column * row_offset, op_trace))
                                    else:
                                        result_heads[head][dim].append(self.load_from_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, row_index + row_offset // rows_per_seq_iteration, 0, self.DRAM_column, op_trace))
                for head in range(self.n_kv_heads):
                    result_head = []
                    for dim in range(self.head_dim):
                        result_head.append(torch.cat(result_heads[head][dim]))
                    result.append(torch.cat(result_head))
        else:
            # print(shape, offset)
            if mode == self.mode["weights"]:
                vector_length = shape[1]
                rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                utilized_banks = (shape[0] - 1) // offset + 1  # shape = [4096, 11008]
                for i in range(utilized_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    rows_per_vector = (vector_length - 1) // self.DRAM_column + 1
                    vectors = []
                    # for vector in range(self.head_dim):
                    if i < utilized_banks - 1:
                        num_vectors = offset
                    else:
                        num_vectors = shape[0] - offset * (utilized_banks - 1)
                    for vector in range(num_vectors):
                        rows = []
                        for row in range(rows_per_vector):
                            if row == rows_per_vector - 1:
                                rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, vector_length - row * self.DRAM_column, op_trace))
                            else:
                                rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_per_vector * vector + row, 0, self.DRAM_column, op_trace))
                        vectors.append(torch.cat(rows))
                    result.append(torch.cat(vectors))
                # print(torch.cat(result).reshape(shape).shape)
            elif mode == self.mode["vector"]:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        dimm_index, channel_index, bank_index = self.bank_index(bank)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif mode == self.mode["score"]:
                for i in range(self.total_banks):
                    dimm_index, channel_index, bank_index = self.bank_index(i)
                    rows = []
                    rows_used = (offset - 1) // self.DRAM_column + 1
                    for row in range(rows_used-1):
                        rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + row, 0, self.DRAM_column, op_trace))
                    rows.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + rows_used - 1, 0, offset - (rows_used - 1) * self.DRAM_column, op_trace))
                    result.append(torch.cat(rows).reshape(-1))
            elif "vector_bank_group" in mode:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                # print(utilized_banks)
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        neighbor_bank_index = int(mode[-1])
                        dimm_index, channel_index, bank_index = self.bank_index(bank*4+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*4+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif "vector_neighbor_bank" in mode:
                utilized_banks = (shape[-1] - 1) // offset + 1  # shape = [1, 1, 4096]
                bank_dim = offset
                bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                result_banks = [[] for _ in range(utilized_banks)]
                for burst in range(bursts_per_bank):
                    for bank in range(utilized_banks):
                        neighbor_bank_index = int(mode[-1])
                        dimm_index, channel_index, bank_index = self.bank_index(bank*2+neighbor_bank_index)
                        if bank < utilized_banks - 1:
                            if burst < bursts_per_bank - 1:
                                num_cols = self.burst_length
                            else:
                                num_cols = bank_dim - burst * self.burst_length
                        else:
                            last_bank_length = shape[-1] - bank * bank_dim
                            last_bank_bursts = (last_bank_length - 1) // self.burst_length + 1
                            if burst < last_bank_bursts - 1:
                                num_cols = self.burst_length
                            elif burst == last_bank_bursts - 1:
                                num_cols = last_bank_length - burst * self.burst_length
                            else:
                                continue
                        result_banks[bank].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, burst * self.burst_length, num_cols, op_trace))
                for bank in range(utilized_banks):
                    result += result_banks[bank]

                # for i in range(utilized_banks):
                #     neighbor_bank_index = int(mode[-1])
                #     dimm_index, channel_index, bank_index = self.bank_index(i*2+neighbor_bank_index)
                #     if i < utilized_banks - 1:
                #         num_cols = offset
                #     else:
                #         num_cols = shape[-1] - offset * (utilized_banks - 1)
                #     result.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index, 0, num_cols, op_trace))
            elif "scores_bank_group" in mode:
                seqlen = offset
                # 4k scores use at most 4 rows
                # store each score in one bank, requring n_head banks
                rows_per_score = (seqlen - 1) // self.DRAM_column + 1
                num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
                bank_group_index = int(mode[-1])
                score_heads = [[] for _ in range(self.n_heads)]
                for k in range(rows_per_score):
                    if k == rows_per_score - 1:
                        bank_dim = seqlen - self.DRAM_column * (rows_per_score - 1)
                    else:
                        bank_dim = self.DRAM_column
                    bursts_per_bank = (bank_dim - 1) // self.burst_length + 1
                    for j in range(num_heads_per_bank):
                        for burst in range(bursts_per_bank):
                            for bank in range(self.total_banks):
                                dimm_index, channel_index, bank_index = self.bank_index(bank)
                                if bank % 4 == bank_group_index:
                                    head = (bank // 4) * num_heads_per_bank + j
                                    if head > self.n_heads - 1:
                                        break
                                    if burst < bursts_per_bank - 1:
                                        num_cols = self.burst_length
                                    else:
                                        num_cols = bank_dim - burst * self.burst_length
                                    score_heads[head].append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, burst * self.burst_length, num_cols, op_trace))
                for head in range(self.n_heads):
                    score = torch.cat(score_heads[head])
                    result.append(score)

                # for i in range(self.total_banks):
                #     dimm_index, channel_index, bank_index = self.bank_index(i)
                #     bank_group_index = int(mode[-1])
                #     if i % 4 == bank_group_index:
                #         for j in range(num_heads_per_bank):
                #             head = (i // 4) * num_heads_per_bank + j
                #             if head > self.n_heads - 1:
                #                 break
                #             scores = []
                #             for k in range(rows_per_score):
                #                 if k == rows_per_score - 1:
                #                     scores.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, offset - k * self.DRAM_column, op_trace))
                #                 else:
                #                     scores.append(self.load_from_DRAM_single_bank(dimm_index, channel_index, bank_index, row_index + j * rows_per_score + k, 0, self.DRAM_column, op_trace))
                #             result.append(torch.cat(scores))
        return torch.cat(result).reshape(shape)