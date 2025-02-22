/*
 * Copyright (c) 2008 Princeton University
 * Copyright (c) 2016 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __MEM_RUBY_NETWORK_GARNET_0_FLITBUFFER_HH__
#define __MEM_RUBY_NETWORK_GARNET_0_FLITBUFFER_HH__

#include <algorithm>
#include <iostream>
#include <vector>

#include "mem/ruby/network/garnet/CommonTypes.hh"
#include "mem/ruby/network/garnet/flit.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

class flitBuffer
{
  public:
    flitBuffer();
    flitBuffer(int maximum_size);

    bool isReady(Tick curTime);
    bool isEmpty();
    void print(std::ostream& out) const;
    bool isFull();
    void setMaxSize(int maximum);
    int getSize() const { return m_buffer.size(); }

    flit *
    getTopFlit()
    {
        flit *f = m_buffer.front();
        m_buffer.pop_front();
        return f;
    }

    flit *
    peekTopFlit()
    {
        return m_buffer.front();
    }

    void
    insert(flit *flt)
    {
        m_buffer.push_back(flt);
    }

    //nghiant: functionalRead now implemented
    //https://www.mail-archive.com/gem5-users@gem5.org/msg19624.html
    bool functionalRead(Packet *pkt);
    //nghiant_end
    bool functionalRead(Packet *pkt, WriteMask &mask);
    uint32_t functionalWrite(Packet *pkt);

  private:
    std::deque<flit *> m_buffer;
    int max_size;
};

inline std::ostream&
operator<<(std::ostream& out, const flitBuffer& obj)
{
    obj.print(out);
    out << std::flush;
    return out;
}

} // namespace garnet
} // namespace ruby
} // namespace gem5

#endif // __MEM_RUBY_NETWORK_GARNET_0_FLITBUFFER_HH__
