// Included after the sealed comparator helpers, before main.
int recover_boundaries(Library& a,Library& b,Totals& totals) {
  int cases=0;
  const bool smoke=std::getenv("RECOVER_SMOKE")!=nullptr;
  auto status=[](Lane& lane,Library& lib,void* plan) {
    fastkpc::StructuredComponentStatus s{};
    check(cudaMemcpyAsync(&s,lib.get(plan).status,sizeof(s),cudaMemcpyDeviceToHost,lane.stream));
    check(cudaStreamSynchronize(lane.stream));return s;
  };
  {
    Lane lane(a,b);
    for(int distinct:{2,3,4,7,8,9,15,16,17,31,32,33,35,36,37,64,127,351}) {
      if(smoke&&distinct!=2&&distinct!=17&&distinct!=351)continue;
      for(int kind=0;kind<4;++kind) {
        if(smoke&&kind!=0&&kind!=3)continue;
        std::vector<double> h(351);
        for(int j=0;j<351;++j) {
          const double x=double((j*97)%distinct)-distinct/2.;
          h[j]=kind==0?x:kind==1?std::ldexp(x,-400):kind==2?std::ldexp(x,400):
            (x==0?(j%2?-0.:0.):x);
        }
        lane.enqueue(h.data(),1);auto r=lane.finish(1)[0];totals.add(r);
        if(r.mask) {
          WorkPairState states[2];int meta[2][10];double inputs[8];
          check(cudaMemcpy(states,a.get(lane.pa).state,sizeof(WorkPairState),cudaMemcpyDeviceToHost));
          check(cudaMemcpy(states+1,b.get(lane.pb).state,sizeof(WorkPairState),cudaMemcpyDeviceToHost));
          check(cudaMemcpy(meta[0],a.get(lane.pa).meta,sizeof(meta[0]),cudaMemcpyDeviceToHost));
          check(cudaMemcpy(meta[1],b.get(lane.pb).meta,sizeof(meta[1]),cudaMemcpyDeviceToHost));
          check(cudaMemcpy(inputs,lane.input,sizeof(inputs),cudaMemcpyDeviceToHost));
          std::cerr<<"boundary failure distinct="<<distinct<<" kind="<<kind<<" input";
          for(double x:inputs)std::cerr<<' '<<x;
          for(int v=0;v<2;++v){std::cerr<<" state"<<v<<" info="<<states[v].info<<" r="<<states[v].r
            <<" count="<<states[v].count<<" canary="<<states[v].canary<<" meta";
            for(int x:meta[v])std::cerr<<' '<<x;}
          std::cerr<<std::endl;
        }
        need(!r.mask,"recover distinct/scale/signed-zero parity");++cases;
      }
    }
    for(int count:{1,2,3,7,8,15,16,17,30,31,32,33,63,64}) {
      if(smoke&&count!=1&&count!=30&&count!=64)continue;
      std::vector<double> h(size_t(count)*351);
      for(int c=0;c<count;++c)for(int j=0;j<351;++j)h[size_t(c)*351+j]=double((j*97)%351)+c*.125;
      lane.enqueue(h.data(),count);
      for(auto r:lane.finish(count)){totals.add(r);need(!r.mask,"recover target-tail parity");}
      ++cases;
    }
  }
  for(int k:{1,7,8,9,15,16,17,31,32,35,36,63,64}) {
    if(smoke&&k!=1&&k!=7&&k!=16&&k!=35&&k!=64)continue;
    Lane lane(a,b,k);std::vector<double> h(351);
    for(int j=0;j<351;++j)h[j]=double((j*97)%351)-175.;
    lane.enqueue(h.data(),1);auto r=lane.finish(1)[0];totals.add(r);
    need(!r.mask,"recover column-tail parity");++cases;
  }
  {
    Lane lane(a,b,35,1);std::vector<double> h(351);
    for(int j=0;j<351;++j)h[j]=double((j*97)%351);
    lane.enqueue(h.data(),1);auto r=lane.finish(1)[0];totals.add(r);
    need(!r.mask,"full pipeline rotation overflow parity");
    auto sa=status(lane,a,lane.pa),sb=status(lane,b,lane.pb);
    need(sa.reason==fastkpc::StructuredRotationOverflow&&sb.reason==sa.reason&&sa.info==sb.info,
         "rotation overflow must reject");++cases;
  }
  {
    Lane lane(a,b);std::vector<double> h(351);
    for(int j=0;j<351;++j)h[j]=double((j*97)%351);
    lane.enqueue(h.data(),1);auto r=lane.finish(1)[0];totals.add(r);need(!r.mask,"fault setup parity");
    auto fa=a.symbol<void(*)(void*)>("rt_recover_fault");
    auto fb=b.symbol<void(*)(void*)>("rt_recover_fault");
    fa(lane.pa);fb(lane.pb);
    auto sa=status(lane,a,lane.pa),sb=status(lane,b,lane.pb);
    need(sa.info==50&&sb.info==50&&sa.reason==fastkpc::StructuredInfo&&sb.reason==sa.reason,
         "recover nonfinite failure must reject");
    // Invalid payloads may be partially written: never compare/publish them.
    ++cases;
  }
  return cases;
}
