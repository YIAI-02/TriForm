sudo ip addr add 192.168.10.2/24 dev eth0 #orange pi
for i in en10 en6 en4 en5; do echo "=== $i ==="; ifconfig $i | grep -i "status\|media\|inet "; done #mac os
sudo ifconfig en10 inet 192.168.10.1 netmask 255.255.255.0 alias #host pc

scp ./README.md root@192.168.10.2:/root/ascend_c_profile
