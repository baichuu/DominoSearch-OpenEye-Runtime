// Whole-model Verilated save/restore at a settled low-clock boundary.
// Trace streams are external to the RTL snapshot and reopened at verified EOF.
#include "Vopeneye_graph_resumable.h"
#include "verilated.h"
#include "verilated_save.h"
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

static std::string option(int argc,char**argv,const std::string& key,const std::string& fallback="") {
    for(int i=1;i<argc;i++)if(argv[i]==key) {
        if(i+1>=argc)throw std::runtime_error("Missing driver argument");return argv[i+1];
    }
    return fallback;
}
static bool flag(int argc,char**argv,const std::string& key) {
    for(int i=1;i<argc;i++)if(argv[i]==key)return true;return false;
}
struct Trace {
    FILE* file;
    explicit Trace(const char* name,bool resume) {
        if(std::filesystem::exists(name)!=resume)throw std::runtime_error("Trace existence differs from resume contract");
        file=std::fopen(name,resume?"a":"w");if(!file)throw std::runtime_error("Cannot open private trace");
        std::setvbuf(file,nullptr,_IOFBF,1<<20);
    }
    void close(){if(file){if(std::fclose(file))throw std::runtime_error("Private trace flush failed");file=nullptr;}}
    ~Trace(){if(file)std::fclose(file);}
};
int main(int argc,char**argv) {
    try {
        const std::string resume=option(argc,argv,"--resume");const bool restoring=!resume.empty();
        const std::string snapshot=option(argc,argv,"--save");
        const uint64_t stop_ip=std::stoull(option(argc,argv,"--stop-ip","4294967295"));
        const uint64_t stop_edges=std::stoull(option(argc,argv,"--stop-after-edges","0"));
        const uint64_t wall_seconds=std::stoull(option(argc,argv,"--wall-seconds","12600"));
        const bool stop_stem=flag(argc,argv,"--stop-stem");
        auto context=std::make_unique<VerilatedContext>();context->commandArgs(argc,argv);
        auto top=std::make_unique<Vopeneye_graph_resumable>(context.get());
        uint64_t time=0;constexpr uint64_t magic=0x444f4d5341564531ULL;
        if(restoring) {
            VerilatedRestore in;in.open(resume.c_str());if(!in.isOpen())throw std::runtime_error("Missing model snapshot");
            uint64_t header=0;in>>header;in>>time;if(header!=magic)throw std::runtime_error("Unknown model snapshot");
            in>>*top;in.close();context->time(time);
            if(top->clk!=0 || !top->rst)throw std::runtime_error("Snapshot is not a settled inference boundary");
        } else {
            top->clk=0;top->rst=1;top->eval();context->timeInc(1000);top->rst=0;top->eval();
            for(int i=0;i<4;i++){context->timeInc(5000);top->clk=1;top->eval();context->timeInc(5000);top->clk=0;top->eval();}
            top->rst=1;top->eval();
        }
        Trace inputs("actual-accepted.hex",restoring),raw("actual-raw.hex",restoring),
              forward("actual-forwarded.hex",restoring),output("actual-output.hex",restoring),stem("actual-stem.hex",restoring);
        const auto start=std::chrono::steady_clock::now();uint64_t edges=0;bool paused=false,budget=false;
        while(!context->gotFinish()) {
            if(top->input_fire_o)std::fprintf(inputs.file,"%016" PRIx64 "\n",uint64_t(top->input_word_o));
            if(top->raw_fire_o)std::fprintf(raw.file,"%016" PRIx64 "\n",uint64_t(top->raw_word_o));
            if(top->forward_fire_o)std::fprintf(forward.file,"%08x %08x %016" PRIx64 "\n",top->instruction_o,top->forward_address_o,uint64_t(top->forward_word_o));
            if(top->final_fire_o)std::fprintf(output.file,"%016" PRIx64 "\n",uint64_t(top->forward_word_o));
            if(top->stem_fire_o)std::fprintf(stem.file,"%02x\n",unsigned(top->stem_word_o));
            context->timeInc(5000);top->clk=1;top->eval();
            context->timeInc(5000);top->clk=0;top->eval();edges++;
            if(context->gotFinish())break;
            const bool target=(stop_stem&&top->stem_boundary_o)||(stop_ip!=0xffffffffULL&&top->last_logged_o==stop_ip);
            budget=edges%65536==0&&std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-start).count()>=wall_seconds;
            if(target || (stop_edges&&edges>=stop_edges) || budget){paused=true;break;}
        }
        inputs.close();raw.close();forward.close();output.close();stem.close();
        if(paused) {
            if(snapshot.empty()||std::filesystem::exists(snapshot))throw std::runtime_error("Fresh private snapshot required");
            VerilatedSave out;out.open(snapshot.c_str());if(!out.isOpen())throw std::runtime_error("Cannot open model snapshot");
            out<<magic;out<<uint64_t(context->time());out<<*top;out.close();
        } else top->final();
        FILE* status=std::fopen("resume-status.json","w");if(!status)throw std::runtime_error("Cannot write private status");
        std::fprintf(status,"{\"status\":\"%s\",\"budget_stop\":%s,\"absolute_cycles\":%" PRIu64 ",\"engine_cycles\":%" PRIu64 ",\"observed_cycles\":%" PRIu64 ",\"simulation_time\":%" PRIu64 ",\"instruction\":%u,\"last_logged_instruction\":%u,\"stem_values\":%u}\n",
            paused?"PAUSED_NOT_NETWORK_PASS":"COMPLETE_REQUIRES_INDEPENDENT_AUDIT",budget?"true":"false",uint64_t(top->cycles_o),uint64_t(top->engine_cycles_o),uint64_t(top->elapsed_o),uint64_t(context->time()),top->instruction_o,top->last_logged_o,top->stem_count_o);
        if(std::fclose(status))throw std::runtime_error("Status flush failed");
        std::printf("CLOCKED_RESUME_DRIVER %s cycles=%" PRIu64 "\n",paused?"PAUSED_NOT_PASS":"COMPLETED_REQUIRES_AUDIT",uint64_t(top->cycles_o));
        return 0;
    } catch(const std::exception&) {std::fprintf(stderr,"Clocked save-restore driver failed\n");return 2;}
}
