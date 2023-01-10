for net in fc rnn
do

    for transfer in no fit env+fit mem+env+fit
    do

        for seed in 0
        do

            if [ $transfer = "no" ] || [ $transfer = "fit" ]; then

                python3 -m gran.rands.ga -n 4 -e envs/multistep/score/control.py \
                                        -b bots/network/static/$net/control.py \
                                        -g 200 -p 16 -f 1 -a '{"task" : "acrobot", "transfer" : "'$transfer'", "seeding" : '$seed'}'
            
            else # [ $transfer = "env+fit" ] || [ $transfer = "mem+env+fit" ]

                python3 -m gran.rands.ga -n 4 -e envs/multistep/score/control.py \
                                        -b bots/network/static/$net/control.py \
                                        -g 200 -p 16 -f 1 -a '{"task" : "acrobot", "transfer" : "'$transfer'", "seeding" : '$seed', "steps" : 100}'

            fi

        done
    done
done

for net in fc rnn
do

    for transfer in no fit env+fit mem+env+fit
    do

        for seed in 0
        do

            if [ $transfer = "no" ] || [ $transfer = "fit" ]; then
                steps=0
            else # [ $transfer = "env+fit" ] || [ $transfer = "mem+env+fit" ]
                steps=100
            fi

            python3 -m gran.rands.evaluate -n 4 \
            -s data/states/envs.multistep.score.control/seeding.$seed~steps.$steps~task.acrobot~transfer.$transfer~trials.1/bots.netted.static.$net.control/16/

        done
    done
done