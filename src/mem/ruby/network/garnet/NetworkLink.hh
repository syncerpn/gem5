/*
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * Copyright (c) 2020 Inria
 * Copyright (c) 2016 Georgia Institute of Technology
 * Copyright (c) 2008 Princeton University
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


#ifndef __MEM_RUBY_NETWORK_GARNET_0_NETWORKLINK_HH__
#define __MEM_RUBY_NETWORK_GARNET_0_NETWORKLINK_HH__

#include <iostream>
#include <vector>

#include "mem/ruby/common/Consumer.hh"
#include "mem/ruby/network/garnet/CommonTypes.hh"
#include "mem/ruby/network/garnet/flitBuffer.hh"
#include "params/NetworkLink.hh"
#include "sim/clocked_object.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

class GarnetNetwork;

class NetworkLink : public ClockedObject, public Consumer
{
  public:
    typedef NetworkLinkParams Params;
    NetworkLink(const Params &p);
    ~NetworkLink() = default;

    void setLinkConsumer(Consumer *consumer);
    void setSourceQueue(flitBuffer *src_queue, ClockedObject *srcClockObject);
    virtual void setVcsPerVnet(uint32_t consumerVcs);
    void setType(link_type type) { m_type = type; }
    link_type getType() { return m_type; }
    void print(std::ostream& out) const {}
    int get_id() const { return m_id; }
    flitBuffer *getBuffer() { return &linkBuffer;}
    virtual void wakeup();

    unsigned int getLinkUtilization() const { return m_link_utilized; }
    const std::vector<unsigned int> & getVcLoad() const { return m_vc_load; }

    inline bool isReady(Tick curTime)
    {
        return linkBuffer.isReady(curTime);
    }

    inline flit* peekLink() { return linkBuffer.peekTopFlit(); }
    inline flit* consumeLink() { return linkBuffer.getTopFlit(); }

    //nghiant: functionalRead now implemented
    //https://www.mail-archive.com/gem5-users@gem5.org/msg19624.html
    bool functionalRead(Packet *pkt);
    //nghiant_end
    bool functionalRead(Packet *pkt, WriteMask &mask);
    uint32_t functionalWrite(Packet *);
    void resetStats();

    std::vector<int> mVnets;
    uint32_t bitWidth;

  private:
    const int m_id;
    link_type m_type;
    const Cycles m_latency;

    ClockedObject *src_object;

    // Statistical variables
    unsigned int m_link_utilized;
    std::vector<unsigned int> m_vc_load;

  protected:
    uint32_t m_virt_nets;
    flitBuffer linkBuffer;
    Consumer *link_consumer;
    flitBuffer *link_srcQueue;

};

} // namespace garnet
} // namespace ruby
} // namespace gem5

#endif // __MEM_RUBY_NETWORK_GARNET_0_NETWORKLINK_HH__
