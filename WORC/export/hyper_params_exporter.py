from WORC.IOparser.config_io_classifier import load_config
from WORC.classification.construct_classifier import create_param_grid
from WORC.classification.trainclassifier import add_parameters_to_grid
from WORC.classification.AdvancedSampler import log_uniform, discrete_uniform, boolean_uniform
from scipy.stats._distn_infrastructure import rv_frozen

printers = {
    log_uniform: lambda x: '$\mathcal{U}^l(' + str(x.base) + '^{' + str(x.loc) + '}, ' + str(x.base) + '^{' + str(
        x.loc + x.scale) + '})$',
    discrete_uniform: lambda x: '$\mathcal{U}^d(' + str(x.loc) + ', ' + str(x.loc + x.scale) + ')$',
    rv_frozen: lambda x: '$\mathcal{U}(' + str(x.kwds['loc']) + ', ' + str(x.kwds['loc'] + x.kwds['scale']) + ')$',
    boolean_uniform: lambda x: '$\mathcal{B}(' + str(x.threshold) + ')$',
    list: lambda x: '{[' + ', '.join([str(y).replace('_', '\_') for y in x]) + ']}'
}

exclude = [  # exclude certain params as they are not part of hyp par optim
    'FeatureScaling_skip_features',
    'FeatPreProcess',
    'OneHotEncoding_feature_labels_tofit',
]

printer_types = tuple(printers.keys())


def export_hyper_params_to_latex(config_file_path, output_file_path):
    config = load_config(config_file_path)
    param_grid = create_param_grid(config)
    params = add_parameters_to_grid(param_grid, config)

    table_out = ''

    for param in sorted(params.keys()):
        if param in exclude:
            continue

        distri = params[param]
        if isinstance(distri, printer_types):
            tex = printers[distri.__class__](distri)
            table_out += param.replace("_", "\\_") + f' & {tex} ' + '\\\\ \\hline\n'
        else:
            raise ValueError(f'Could not map {param} - {distri.__dict__}')

    table = """\\begin{table}[]
\\begin{tabular}{l|l}
""" + table_out + """
\\end{tabular}
\\end{table}
"""

    with open(output_file_path, 'w') as fh:
        fh.write(table)
