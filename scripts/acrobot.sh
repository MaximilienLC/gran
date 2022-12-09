for net in fc rnn
do

    for transfer in no fit env+fit mem+env+fit
    do

        for seed in 0 1 2 3 4 5 6 7 8 9
        do

            python3 -m gran.rands.ga -n 4 -e envs/multistep/score/control.py \
                                    -b bots/network/static/$net/control.py \
                                    -g 100 -p 16 -a '{"task" : "acrobot", "transfer" : "'$transfer'", "seeding" : '$seed'}'

        done
    done
done

for net in fc rnn
do

    for transfer in no fit env+fit mem+env+fit
    do

        for seed in 0 1 2 3 4 5 6 7 8 9
        do

            python3 -m gran.rands.evaluate -n 4 \
            -p data/states/envs.multistep.score.control/seeding.$seed~steps.0~task.acrobot~transfer.$transfer~trials.1/bots.network.static.$net.control/16/
            -s $seed

        done
    done
done