def get_layer_stack(workload_name:str):
    # the predefined layer stack for each workload
    if workload_name == "resnet18":
        layer_stacks = [tuple(range(0, 11)),
                        tuple(range(11, 22)),
                        tuple(range(22, 28)),
                        tuple(range(28, 33)),
                        tuple(range(33, 39)),
                        tuple(range(39, 45))
                        ] + list((i,) for i in range(45, 49))
    if workload_name == "fsrcnn":
        layer_stacks = list((i,) for i in range(0, 8))
        # layer_stacks = [tuple(range(0,8))]
    if workload_name == "squeezenet":
        layer_stacks = [tuple(range(0, 9)),
                        tuple(range(9, 16)),
                        tuple(range(16, 24)),
                        tuple(range(24, 31)),
                        tuple(range(31, 39)),
                        tuple(range(39, 46)),
                        tuple(range(46, 53)),
                        tuple(range(53, 60))
                        ] + list((i,) for i in range(60, 68))
    if workload_name == "mobilebert":
        layer_stacks = None
    if workload_name == "tinyyolo2":
        layer_stacks = None
    if workload_name == "xception":
        layer_stacks = None
    if workload_name == "mobilenetv2":
        layer_stacks = [tuple(range(0, 10)),
                        tuple(range(10, 16)),
                        tuple(range(16, 27)),
                        tuple(range(27, 33)),
                        tuple(range(33, 43)),
                        tuple(range(43, 50)),
                        tuple(range(56, 67)),
                        tuple(range(67, 73)),
                        tuple(range(73, 84)),
                        tuple(range(84, 90))
                        ] + list((i,) for i in range(90, 100))
    if workload_name == "inception_v2":
        layer_stacks = [tuple(range(0, 6)),
                        tuple(range(6, 14)),
                        tuple(range(14, 21)),
                        tuple(range(21, 23)),
                        tuple(range(23, 30)),
                        tuple(range(30, 36)),
                        tuple(range(36, 44)),
                        tuple(range(44, 50)),
                        tuple(range(50, 57)),
                        tuple(range(57, 64)),
                        tuple(range(65, 73)),
                        tuple((73,)),
                        tuple(range(74, 81)),
                        tuple(range(81, 88)),
                        tuple(range(88, 95)),
                        tuple(range(95, 102)),
                        tuple(range(102, 109)),
                        tuple(range(109, 116)),
                        tuple(range(116, 124)),
                        tuple((124,)),
                        tuple(range(125, 132)),
                        tuple(range(132, 139)),

                        tuple(range(139, 146)),
                        tuple(range(146, 153)),
                        tuple(range(153, 160)),
                        tuple((160,)),
                        tuple((161,)),
                        tuple(range(162, 169)),
                        tuple(range(169, 176)),
                        tuple(range(176, 183)),
                        tuple(range(183, 190)),
                        tuple(range(190, 197)),
                        tuple(range(197, 204)),
                        tuple(range(204, 212)),
                        tuple((212,)),
                        tuple(range(213, 220)),
                        tuple(range(220, 227)),
                        tuple(range(227, 234)),
                        tuple(range(234, 241)),
                        tuple(range(241, 248)),
                        tuple(range(248, 255)),
                        tuple(range(255, 263)),
                        tuple((263,)),
                        tuple(range(264, 271)),
                        tuple(range(271, 278)),
                        tuple(range(278, 285)),
                        tuple(range(285, 292)),
                        tuple(range(292, 299)),
                        tuple(range(299, 306)),
                        tuple(range(306, 314)),
                        tuple((314,)),
                        tuple(range(315, 322)),
                        tuple(range(322, 329)),
                        tuple(range(329, 336)),
                        tuple(range(336, 343)),
                        tuple(range(343, 350)),
                        tuple(range(350, 357)),
                        tuple(range(357, 365)),
                        tuple((365,)),    
                        tuple(range(366, 373)),
                        tuple(range(373, 380)),
                        tuple(range(380, 387)),
                        tuple(range(387, 394)),
                        tuple(range(394, 401)),
                        tuple((401,)), 
                        tuple((402,)),

                        tuple(range(403, 410)),
                        tuple(range(410, 417)),
                        tuple(range(417, 424)),
                        tuple(range(424, 431)),
                        tuple(range(431, 438)),
                        tuple(range(438, 445)),
                        tuple(range(445, 453)),
                        tuple((453,)),
                        tuple(range(454, 461)),
                        tuple(range(461, 468)),
                        tuple(range(468, 475)),
                        tuple(range(475, 482)),
                        tuple(range(482, 489)),
                        tuple(range(489, 496)),
                        tuple(range(496, 504))
                        ] + list((i,) for i in range(504, 509))
             
    return layer_stacks

