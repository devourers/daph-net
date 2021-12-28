import count
import segmentation
import json


def main(path, model_path, real_data=None):
    inp = 0
    model = segmentation.Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    seg_res = segmentation.process_image_sequence(model, inp)
    json_res = []
    count_res = []
    for img in seg_res:
        c_r, m_r = count.count_and_form_markup(img)
        json_res.append(m_r)
        count_res(c_r)
    json.dump(json_res)
    #plot image