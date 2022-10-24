//
// Created by xiewanpeng on 2022/10/24.
//

#ifndef DISTRIBUTED_LINEARMODEL_CMDS_H
#define DISTRIBUTED_LINEARMODEL_CMDS_H

namespace dist_linear_model {

    enum MODELCOMMANDS {
        UNKNOWN = 1,
        TRAIN = 2,
        TEST = 3,
        LOAD = 4,
        LOADINC = 5,
        SAVE = 6,
    };

}

#endif //DISTRIBUTED_LINEARMODEL_CMDS_H
