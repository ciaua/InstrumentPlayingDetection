import SharedArray as sa


def make_array(array, arr_name):
    name = 'shm://{}'.format(arr_name)
    new_array = sa.create(name=name, shape=array.shape, dtype=array.dtype)
    new_array[:] = array

    return new_array, name


def make_array_noreturn(array, arr_name):
    name = 'shm://{}'.format(arr_name)
    new_array = sa.create(name=name, shape=array.shape, dtype=array.dtype)
    new_array[:] = array


def get_array(name):
    try:
        array = sa.attach(name)
    except OSError:
        array = None

    return array


def delete_array(name):
    sa.delete(name)


def delete_all(prefix='jy'):
    for term in sa.list():
        name = term.name
        if name.startswith('{}.'.format(prefix)):
            delete_array(name)


def array_in_list(name):
    for arr in sa.list():
        if name == arr.name:
            return True

    return False
