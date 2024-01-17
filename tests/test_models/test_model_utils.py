from sign_language_translator.models.utils import (
    FullyLambdaLR,
    downwards_wave,
    plot_lr_scheduler,
    top_p_top_k_indexes,
)


def test_top_p_top_k_indexes():
    probs = [0.1, 0.2, 0.15, 0.05, 0.3, 0.2]
    top_p = 0.75
    top_k = 3

    assert top_p_top_k_indexes(probs, top_p, top_k) == [4, 1, 5]
    assert top_p_top_k_indexes(probs, None, None) == [4, 1, 5, 2, 0, 3]


def test_lr_scheduling():
    wave = downwards_wave(5, 7, 0.01, 1e-5)

    plot_lr_scheduler(
        lr_scheduler_class=FullyLambdaLR,
        initial_lr=0.01,
        n_steps=len(wave) - 1,
        lr_lambda=(lambda step_num, base_lr, last_lr: wave[step_num]),
        save_fig=True,
        fig_name="temp_wave.png",
    )
