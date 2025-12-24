sudo ip addr add 192.168.10.2/24 dev eth0 #orange pi
for i in en10 en6 en4 en5; do echo "=== $i ==="; ifconfig $i | grep -i "status\|media\|inet "; done #mac os
sudo ifconfig en10 inet 192.168.10.1 netmask 255.255.255.0 alias #host pc

scp ./README.md root@192.168.10.2:/root/ascend_c_profile

nmcli dev wifi
sudo nmcli dev wifi connect OPPO password yjq021212

sed -i 's/\r$//' run.sh

cp -r ./22_baremix_kernellaunch ../Test

tar czf sim_out.tar.gz simulator/

bash run.sh -r sim -v Ascend310B1
export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest


find . -type f -name 'core0.veccore0_code_exe_*.csv' -exec cp -vn {} ../../fit_model/softmax/results/ \;