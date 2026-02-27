import numpy as np, sys
from ft8_decode import ft8_unpack_message, _unpack_grid, _bits_to_int

def ibits(v, l): return [(v >> (l-1-i)) & 1 for i in range(l)]

NBASE=37*36*10*27*27*27
C36='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
C37=' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
C27='ABCDEFGHIJKLMNOPQRSTUVWXYZ '; C10='0123456789'

def pk(call):
    call=call.upper().strip()
    if call=='CQ': return NBASE+2
    dp=next((i for i,c in enumerate(call) if c.isdigit()), -1)
    if dp<0: pre,d,suf='','0',call[:3]
    else: pre,d,suf=call[:dp],call[dp],call[dp+1:]
    pr2=(pre+'  ')[:2]; s3=(suf+'   ')[:3]; c=list(pr2+d+s3)
    n=C36.index(c[0]); n=n*37+C37.index(c[1]); n=n*10+C10.index(c[2])
    n=n*27+C27.index(c[3]); n=n*27+C27.index(c[4]); n=n*27+C27.index(c[5]); return n

c1=pk('W4ABC'); c2=pk('K9XYZ'); g73=32765
print(f"c1={c1}, c2={c2}, g73={g73}")
print(f"_unpack_grid(32404, 0) = {_unpack_grid(32404, 0)}")  # 73

bits = np.array(ibits(c1,28)+[0]+ibits(c2,28)+[0]+ibits(g73,15)+[0]+ibits(0,3), dtype=np.uint8)
print(f"bits length = {len(bits)}")

gval = _bits_to_int(bits, 58, 15)
print(f"grid bits[58:73] = {gval}")
i3 = _bits_to_int(bits, 74, 3)
n3 = _bits_to_int(bits, 71, 3)
print(f"i3={i3}, n3={n3}")

result = ft8_unpack_message(bits)
print(f"ft8_unpack_message = '{result}'")
sys.stdout.flush()

