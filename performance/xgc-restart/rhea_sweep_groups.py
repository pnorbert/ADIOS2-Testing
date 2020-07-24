from sweep_groups_helper import create_sweep_groups
# Parameters 64, 256, 1024 nodes  16 processes per node
writer_np               = [1024, 4096, 16384]
reader_np_ratio         = [1]
writers_per_node        = [16]
size_per_pe             = []
engines                 = ['bp4']
run_repetitions         = 0
batch_job_timeout_secs  = 3600
per_experiment_timeout  = 600

node_layouts = []
node_layouts.append([ {'writer': writers_per_node[0]}, {'reader': writers_per_node[0]}])
sweep_groups = create_sweep_groups ('rhea',
                                    writer_np,
                                    reader_np_ratio,
                                    size_per_pe,
                                    engines,
                                    node_layouts,
                                    run_repetitions,
                                    batch_job_timeout_secs,
                                    per_experiment_timeout)