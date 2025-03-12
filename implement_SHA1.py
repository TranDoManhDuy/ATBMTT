import struct

def left_rotate(n, b):
    return ((n << b) | (n >> (32 - b))) & 0xffffffff

def sha1_implement(message):
    # Chuẩn bị thông điệp
    orig_len_in_bits = (8 * len(message)) & 0xffffffffffffffff
    message += b'\x80'
    while (len(message) * 8) % 512 != 448:
        message += b'\x00'
    message += struct.pack('>Q', orig_len_in_bits)
    
    # Khởi tạo các hằng số
    h0, h1, h2, h3, h4 = (
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    )
    
    # Xử lý từng khối 512-bit
    # 64 bytes => 512 bits
    for i in range(0, len(message), 64):
        w = list(struct.unpack('>16I', message[i:i+64])) + [0] * 64
        for j in range(16, 80):
            w[j] = left_rotate(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1)
        
        a, b, c, d, e = h0, h1, h2, h3, h4
        
        for j in range(80):
            if j < 20:
                f = (b & c) | ((~b) & d)
                k = 0x5A827999
            elif j < 40: 
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif j < 60:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            else:
                f = b ^ c ^ d
                k = 0xCA62C1D6
            
            temp = (left_rotate(a, 5) + f + e + k + w[j]) & 0xffffffff
            e = d
            d = c
            c = left_rotate(b, 30)
            b = a
            a = temp
        h0 = (h0 + a) & 0xffffffff
        h1 = (h1 + b) & 0xffffffff
        h2 = (h2 + c) & 0xffffffff
        h3 = (h3 + d) & 0xffffffff
        h4 = (h4 + e) & 0xffffffff

    return ''.join(format(x, '08x') for x in (h0, h1, h2, h3, h4))

def sha1_file(filename):
    content = b""
    with open(filename, 'rb') as f:
        while chunk := f.read(4096):
            content += chunk
    return sha1_implement(content)

# print("SHA-1 của file:", sha1_file("shattered-1.pdf"))
# print("SHA-2 của file:", sha1_file("shattered-2.pdf"))