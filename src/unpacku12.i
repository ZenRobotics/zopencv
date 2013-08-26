
%{

void cvZUnpackU12 (int npairs, const char* _input, char* _output) {
    const unsigned char* input =
            reinterpret_cast<const unsigned char*> (_input);
    unsigned short* output = reinterpret_cast<unsigned short*>(_output);
    for (int i = 0; i < npairs; ++i) {
        unsigned char a, b, c;
        unsigned short x, y;
        a = *input++;
        b = *input++;
        c = *input++;
        x = (a << 4) | (b & 0xf);
        y = (c << 4) | (b >> 4);
        *output++ = x;
        *output++ = y;
    }
}
%}

// Unpack basler unsigned 12-bit data. Second argument is input data, third
// output. Input is basler's Mono12 format, see
// smb://samba/hardware/vision/basler section 10.2.3 Mono 12 Packed Format
// (count npairs * 2). Output is coerced into 16-bit
// and each 12-bit data point unpacked into one 16-bit value in it.
void cvZUnpackU12 (int npairs, const char* BYTE, char* BYTE);
