export AWS_PROFILE=softmax

# list jobs
alias jj="sky jobs queue --skip-finished" # avoid conflict with jq (JSON processor)
alias jja="sky jobs queue"

# cancel ("kill") job
alias jk="sky jobs cancel -y"
alias jka="sky jobs cancel -a -y"

# get logs
alias jl="sky jobs logs"
alias jlc="sky jobs logs --controller"
alias jll='sky jobs logs $(jj | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'
alias jllc='sky jobs logs --controller $(jj | grep -A1 TASK | grep -v TASK | awk "{print \$1}")'

# launch training
alias lt="./devops/skypilot/launch.py train"
