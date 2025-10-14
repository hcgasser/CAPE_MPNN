from datetime import datetime


def now(formate="%Y-%m-%d %H:%M:%S", return_obj=False):
    """returns the current time in the given format

    :param formate: format of the time string (default: '%Y-%m-%d %H:%M:%S')
    :return: current time
    """

    jetzt = datetime.now()
    if return_obj:
        return jetzt.strftime(formate), jetzt
    return jetzt.strftime(formate)
