import re



def rnormalize(name):
    return re.sub(r"(.+?)(\(|\[).+",lambda x:x.group(1), name)


def normalize(name):
    import re
    return re.sub(r"(\(|\[).+?(\)|\])","", name.strip())


def normalize_protein(name):
    '''

    :param name: Chain L, Capsid protein;envelope polyprotein, partial;Tyrosine-protein kinase transforming protein Src
    :return:
    '''
    name = normalize(name)
    # name = name.lower().strip()
    if "Chain" in name and "," in name:
        return name.split(",", maxsplit=1)[1].strip()
    elif name.endswith(", partial"):
        return name.rsplit(",", maxsplit=1)[0].strip()
    elif name.lower().endswith(" src"):
        return name[:-4]
    # if "protein " in name:
    #     index = name.index("protein ")
    #     name=name[:index+7]
    return name if len(name.strip()) > 7 else ""


def normalize_virus(name):
    '''

    :param name: Human alphaherpesvirus 1 strain KOS
    :return:
    '''
    name = normalize(name)
    name = name.lower().strip()
    if "virus " in name:
        index = name.index("virus ")
        return name[:index+5]
    return name