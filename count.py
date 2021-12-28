def count(image):
    raise NotImplementedError


def form_markup(count_result):
    raise NotImplementedError


def count_and_form_markup(image):
    c_r = count(image)
    m_r = form_markup(c_r)
    return c_r, m_r