import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.skipif(__import__('shutil').which('ffmpeg') is None, reason='ffmpeg is needed')
def test_a_clip_gives_one_answer_per_layer_of_cortex_through_participate(tmp_path):
    import subprocess
    import pandas as pd
    from brainscore import load_model
    from brainscore_core.streaming_helpers import neural_response
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    clip = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=240x224:rate=60:duration=1.49',
                    str(clip)], check=True)
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a', 'b'], 'video_path': [str(clip)] * 2}))
    stimuli.identifier = 'made-up-mario-clip'
    response = neural_response(load_model('paugam2025-untrained-cnn'), stimuli, ['neural:cortex'])
    layers = np.asarray(response['layer'].values).astype(str)
    assert sorted(set(layers)) == ['base.conv2', 'base.conv3', 'base.conv4', 'base.linear']
    assert (layers == 'base.linear').sum() == 512
    values = np.asarray(response.values).reshape(2, -1)
    assert np.allclose(values[0], values[1])          # the same clip, the same answer: no state, fixed weights


def test_the_weights_are_drawn_once_and_the_same_each_time():
    from brainscore.models.paugam2025_untrained_cnn.model import build_network
    first, second = build_network(), build_network()
    assert all(np.array_equal(a.detach().numpy(), b.detach().numpy())
               for a, b in zip(first.parameters(), second.parameters()))


def test_the_input_is_the_papers_stack_of_four_grey_frames():
    from brainscore.models.paugam2025_untrained_cnn.model import preprocess
    frames = [np.full((224, 240, 3), value, np.uint8) for value in range(16)]
    stack = preprocess(frames)
    assert tuple(stack.shape) == (1, 4, 84, 84)
    assert stack[0, :, 0, 0].tolist() == pytest.approx([v / 255 for v in (3, 7, 11, 15)])
