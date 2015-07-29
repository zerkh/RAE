#! /bin/bash
iconv -f GBK -t UTF-8 $* > u
mv u $*
