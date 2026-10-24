"""Research-only patch: in the persistent optimizer batch, wait for the setup's own
stream (blocking-sync event, thread sleeps) before the pageable status D2H, so the
pageable D2H path is never held across a running kernel. Numerics unchanged."""
import sys
path = sys.argv[1]
src = open(path).read()
old = '''      "record persistent multi-penalty residual completion");
    check_cuda(cudaMemcpyAsync(
      prepared->host_states.data(), prepared->d_states.get(),'''
new = '''      "record persistent multi-penalty residual completion");
    {
      struct HostWaitEvent {
        cudaEvent_t event = nullptr;
        ~HostWaitEvent() { if (event != nullptr) cudaEventDestroy(event); }
      };
      static thread_local HostWaitEvent host_wait;
      if (host_wait.event == nullptr) {
        check_cuda(cudaEventCreateWithFlags(
          &host_wait.event, cudaEventDisableTiming | cudaEventBlockingSync),
          "create research host wait event");
      }
      check_cuda(cudaEventRecord(host_wait.event, prepared->stream.get()),
                 "record research host wait event");
      check_cuda(cudaEventSynchronize(host_wait.event),
                 "wait research host wait event");
    }
    check_cuda(cudaMemcpyAsync(
      prepared->host_states.data(), prepared->d_states.get(),'''
assert src.count(old) == 1, src.count(old)
open(path, 'w').write(src.replace(old, new))
print('patched', path)
