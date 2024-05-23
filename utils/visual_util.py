import os
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_TSinterpret(exp_model, orig, pred, CF, pred_CF, path, marker):
    if not os.path.exists(f'{path}/fig'):
        os.makedirs(f'{path}/fig')
    exp_model.plot_in_one(orig, pred, CF, pred_CF,
                          save_fig=f'{path}/fig/{marker}_2in1_pre{pred}CF{pred_CF}.png')
    exp_model.plot(orig, pred, CF, pred_CF,
                   save_fig=f'{path}/fig/{marker}_sep_pre{pred}CF{pred_CF}')
