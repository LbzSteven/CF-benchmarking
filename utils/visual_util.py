import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets

from utils.data_util import get_valid_CF_given_path, read_UCR_UEA


def plot(original, org_label, exp, exp_label, vis_change=True, all_in_one=False, save_fig=None, figsize=(6.4, 4.8)):
    """
    Basic Plot Function for visualizing counterfactuals.
    Arguments:
        original np.array: Instance to be explained.
        org_label int: Label of instance to be explained.
        exp np.array: Explanation.
        exp_label int: Label of Explanation.
        vis_change bool: Change to be visualized as heatmap.
        all_in_one bool: Original and Counterfactual in one plot.
        save_fig str: Path to save fig at.
    """
    plt.figure(figsize=figsize)
    if all_in_one:
        ax011 = plt.subplot(1, 1, 1)
        ax012 = ax011.twinx()
        sal_02 = np.abs(original.reshape(-1) - np.array(exp).reshape(-1)).reshape(
            1, -1
        )
        if vis_change:
            sns.heatmap(
                sal_02,
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=ax011,
                yticklabels=False,
            )
        else:
            sns.heatmap(
                np.zeros_like(sal_02),
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=ax011,
                yticklabels=False,
            )
        sns.lineplot(
            x=range(0, len(original.reshape(-1))),
            y=original.flatten(),
            color="white",
            ax=ax012,
            legend=False,
            label="Original",
        )
        sns.lineplot(
            x=range(0, len(original.reshape(-1))),
            y=exp.flatten(),
            color="black",
            ax=ax012,
            legend=False,
            label="Counterfactual",
        )
        plt.legend()

    else:
        ax011 = plt.subplot(2, 1, 1)
        ax012 = ax011.twinx()
        sal_02 = np.abs(original.reshape(-1) - np.array(exp).reshape(-1)).reshape(
            1, -1
        )
        if vis_change:
            sns.heatmap(
                sal_02,
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=ax011,
                yticklabels=False,
            )
        else:
            sns.heatmap(
                np.zeros_like(sal_02),
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=ax011,
                yticklabels=False,
            )

        p = sns.lineplot(
            x=range(0, len(original.reshape(-1))),
            y=original.flatten(),
            color="white",
            ax=ax012,
            label=f"{org_label}",
        )
        p.set_ylabel("Original")

        ax031 = plt.subplot(2, 1, 2)
        ax032 = ax031.twinx()
        sal_02 = np.abs(original.reshape(-1) - np.array(exp).reshape(-1)).reshape(
            1, -1
        )
        if vis_change:
            sns.heatmap(
                sal_02,
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=ax031,
                yticklabels=False,
            )
        else:
            sns.heatmap(
                np.zeros_like(sal_02),
                fmt="g",
                cmap="viridis",
                cbar=False,
                ax=ax011,
                yticklabels=False,
            )

        p = sns.lineplot(
            x=range(0, len(original.reshape(-1))),
            y=exp.flatten(),
            color="white",
            ax=ax032,
            label=f"{exp_label}",
        )
        p.set_ylabel("Counterfactual")
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig,dpi=300)


def plot_in_one(
    item, org_label, exp, cf_label, save_fig=None, figsize=(6.4, 4.8)
):

    item = item.reshape(item.shape[-2], item.shape[-1])
    exp = exp.reshape(item.shape[-2], item.shape[-1])
    ind = ""
    # print("Item Shape", item.shape[-2])
    if item.shape[-2] > 1:
        res = (item != exp).any(-1)
        # print(res)
        ind = np.where(res)
        # print(ind)
        if len(ind[0]) == 0:
            print("Items are identical")
            return
        else:
            item = item[ind]
            exp = exp[ind]

    plt.style.use("classic")
    colors = [
        "#08F7FE",  # teal/cyan
        "#FE53BB",  # pink
        "#F5D300",  # yellow
        "#00ff41",  # matrix green
    ]
    df = pd.DataFrame(
        {
            f"Predict: {org_label[0]}": list(item.flatten()),
            f"CF: {cf_label[0]}": list(exp.flatten()),
        }
    )
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(marker=".", color=colors, ax=ax).legend(bbox_to_anchor=(1.0, 1.0),fontsize='small',)
    # Redraw the data with low alpha and slighty increased linewidth:
    n_shades = 10
    diff_linewidth = 1.05
    alpha_value = 0.3 / n_shades
    for n in range(1, n_shades + 1):
        df.plot(marker=".", linewidth=2 + (diff_linewidth * n), alpha=alpha_value, legend=False, ax=ax, color=colors)
    ax.grid(color="#2A3459")
    plt.xlabel("Time", fontweight="bold", fontsize="large")

    if ind != "":
        plt.ylabel(f"Feature {ind[0][0]}", fontweight="bold", fontsize="large")
    else:
        plt.ylabel("Value", fontweight="bold", fontsize="large")
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig,dpi=800)

def visualize_TSinterpret(orig, pred, CF, pred_CF, path, marker,save=False,save_path = None):
    if not os.path.exists(f'{path}/fig'):
        os.makedirs(f'{path}/fig')
    in_one_save = None if not save else f'{path}/fig/{marker}_2in1_pre{pred}CF{pred_CF}.png' if save_path is None else save_path
    # plot_save = None if not save else f'{path}/fig/{marker}_sep_pre{pred}CF{pred_CF}'
    plot_in_one(orig, pred, CF, pred_CF, save_fig=in_one_save)
    plot(orig, pred, CF, pred_CF, save_fig=None)


def plot_given_valid_cfs(num, CF_result, name,save,save_path):
    valid,tx_valid,ty_valid,pred_valid,cf,cf_pred,random_selection, num_instance = CF_result
    if 'wCF' in name or 'TSEvo' in name:
        checker = random_selection[valid]
    else:
        checker = valid
    if num in checker:
        pos = np.where(checker==num)[0]
        orig = tx_valid[pos]
        pred_label = pred_valid[pos]
        CF = cf[pos]
        pred_CF= cf_pred[pos]
        CF_path = None
        visualize_TSinterpret(orig, pred_label, CF, pred_CF, CF_path, marker='', save=save,save_path=save_path)
    else:
        print(f'{name} doenst have CF for instance {num}')

def given_dataset_model_visualize_CFs(dataset, model, CF_names=None, num=None, save=False,save_dir=None):
    if CF_names is None:
        CF_names = ['NG', 'NUN_CF', 'TSEvo', 'SETS', 'wCF']
    def find_common_elements(lists):
        common_elements = set(lists[0])

        for sublist in lists[1:]:
            temp = common_elements.intersection(sublist)
            if len(temp) == 0:
                print('No mutual results')
                return list(common_elements)
            else:
                common_elements = temp
        return list(common_elements)
    valids = []
    CF_results ={}
    for CF_name in CF_names:
        CF_path = f'./CF_result/{CF_name}/{model}/{dataset}/'
        CF_result = get_valid_CF_given_path(CF_path)
        valid, tx_valid, ty_valid, pred_valid, cf, cf_pred, random_selection = CF_result
        CF_results[CF_name] = CF_result
        if 'wCF' in CF_name or 'TSEvo' in CF_name:
            checker = random_selection[valid]
        else:
            checker = valid
        if 'wCF' in CF_name or 'TSEvo' in CF_name or 'SETS' in CF_name:
            print(f"{CF_name} valid instance: {checker}")
        valids.append(checker)
    common_elements = find_common_elements(valids)
    print(common_elements)
    if num is None:
        num = common_elements[0]
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for CF_name, CF_result in CF_results.items():
        print(f'{CF_name} with {num}')

        save_path = os.path.join(save_dir, f'{CF_name}_inst{num}.png') if save else None
        plot_given_valid_cfs(num, CF_result, CF_name,save,save_path)
    return CF_results


def show_shapelets(dataset, class_index=None, save=True, save_dir=None):
    shapelets_path = f'./shapelets/{dataset}/SETS.pkl'
    if os.path.exists(shapelets_path):
        with open(shapelets_path, 'rb') as file:
            exp_model = pickle.load(file)
    print(exp_model.all_shapelets_class)
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_datasets())

    fittedshaplets = exp_model.fitted_shapelets.copy()
    all_heat_maps = exp_model.all_heat_maps
    if class_index is None:
        for key, value in exp_model.all_shapelets_class.items():

            if len(value[0])>0:
                class_index=key
                break
    print(class_index)
    class_dim0_heatmap = all_heat_maps[class_index][0]
    print(all_heat_maps.keys())

    instance = np.where(np.argmax(train_y, axis=1) == class_index)[0][0]

    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in class_dim0_heatmap.keys():
        shapelet_i = fittedshaplets[0][i]
        current_heat = class_dim0_heatmap[i]
        starting = np.argwhere(current_heat > 0)[0][0]
        plt.figure()
        plt.plot(range(shapelet_i.shape[1]) + starting, np.squeeze(shapelet_i),
                 label=f'shapelet:{i},shape:{shapelet_i.shape}')
        plt.plot(np.squeeze(train_x[instance]), label=f'train instance {instance}')
        plt.legend()

        if save:
            save_path = os.path.join(save_dir, f'class{class_index}_shapelet{i}.png')
            plt.savefig(save_path, dpi=300)

    for key, class_heatmaps in all_heat_maps.items():
        class_heatmap_0 = class_heatmaps[0]
        for i in class_heatmap_0.keys():
            print(f'class{key},shapelet:{i}, length{fittedshaplets[0][i].shape[1]}')

def show_instance_class(dataset, save=True,save_dir = None,i=0):
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=UCR_UEA_datasets())
    train_y_orig = np.argmax(train_y,axis=1)
    input_size = train_x.shape[-1]
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure()
    for class_index in np.unique(train_y_orig):
        instance_index = np.where(train_y_orig==class_index)[0][0]
        plt.plot(range(input_size),train_x[instance_index].flatten(),label=f'instance {instance_index} label:{train_y_orig[instance_index]}')
    plt.legend()
    if save:
        save_path = os.path.join(save_dir, f'instance_pair{i}.png')
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    return train_x, test_x, train_y, test_y,input_size