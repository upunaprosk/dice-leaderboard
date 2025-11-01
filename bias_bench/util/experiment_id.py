def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
    quant_type=None,
    quant_prec=None,
    revision=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        model = model.replace('/', '_')
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        model_name_or_path = model_name_or_path.replace('/', '_')
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(quant_type, str):
        experiment_id += f"_qt-{quant_type}"
    if isinstance(quant_prec, str):
        experiment_id += f"_qp-{quant_prec}"
    if isinstance(revision, str):
        experiment_id += f"_rev-{revision}"

    # filter out pythia extra strs
    experiment_id = experiment_id.replace('EleutherAI_', '')
    experiment_id = experiment_id.replace('EleutherAI-', '')
    experiment_id = experiment_id.replace('pythia_', '')
    experiment_id = experiment_id.replace('pythia-', '')
    experiment_id = experiment_id.replace('-deduped', '')
    experiment_id = experiment_id.replace('_deduped', '')
    experiment_id = experiment_id.replace('step', '')

    return experiment_id
