##########################################################################
## Module Info
##########################################################################

__version_info__ = {
    "major": 0,
    "minor": 2,
    "micro": 0,
    "releaselevel": "final",
    "serial": 1,
}

##########################################################################
## Helper Functions
##########################################################################


def get_version(short=False):
    """
    Returns the version string for prompt-logger.
    """
    assert __version_info__["releaselevel"] in ("alpha", "beta", "final")
    vers = [
        "%(major)i.%(minor)i" % __version_info__,
    ]
    if __version_info__["micro"]:
        vers.append(".%(micro)i" % __version_info__)
    if __version_info__["releaselevel"] != "final" and not short:
        vers.append(
            "%s%i" % (__version_info__["releaselevel"][0], __version_info__["serial"])
        )
    return "".join(vers)
