const std = @import("std");
const args = @import("args");
const ptk = @import("ptk");
const builtin = @import("builtin");
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    var ally = gpa.allocator();
    // The great normalizer
    // If the string is first in the line and have a colon, it is a label
    // if it's after a "=" it's a macro string
    // if it's after a "@" it's a macro name
    // if it's after a ";" it's a comment
    // We avoid normalizing these strings
    var file = try std.fs.cwd().openFile("script.fur", .{});
    var example_program = try file.readToEndAlloc(ally, std.math.maxInt(usize));
    defer file.close();
    defer ally.free(example_program);

    var line_it = std.mem.tokenize(u8, example_program, "\n\r");
    // if it starts in ; we copy the line
    // if it starts in @ we copy the line
    // otherwise we lowercase the line
    var p = arr: {
        var program = std.ArrayList(u8).init(ally);
        defer program.deinit();

        var arena = std.heap.ArenaAllocator.init(ally);
        defer arena.deinit();

        while (line_it.next()) |line_untrimmed| {
            var line = std.mem.trim(u8, line_untrimmed, " \t");
            if (line.len == 0) {
                continue;
            }
            if (line[0] == ';') {
                try program.appendSlice(line);
                try program.appendSlice("\n");
                continue;
            }
            if (line[0] == '@') {
                try program.appendSlice(line);
                try program.appendSlice("\n");
                continue;
            }
            var colon = std.mem.indexOf(u8, line, ":");
            if (colon == null) {
                try program.appendSlice(try std.ascii.allocLowerString(arena.allocator(), line));
                try program.appendSlice("\n");
                continue;
            }
            var label = line[0..colon.?];
            var rest = line[colon.? + 1 ..];
            try program.appendSlice(label);
            try program.appendSlice(":");
            try program.appendSlice(rest);
            try program.appendSlice("\n");
        }
        break :arr try program.toOwnedSlice();
    };
    var parser = try FurAsm.initBuffer(ally, std.mem.trim(u8, p, " \t"));
    defer parser.deinit();
    try parser.run();
}

/// We implement FurASM a esolang with FASM like qualities
/// https://esolangs.org/wiki/FurASM
/// Rules:
/// A line marks an instruction if it is not a comment or macro.
/// Instruction arguments are separated by whitespace.
/// Instruction arguments can only be register identifiers or int literals.
/// A line marks a comment if it starts with an at sign (@).
/// If a macro contains an equals sign (=), any text after it and before a newline (\n) is considered a macro value.
/// Anything after a semicolon (;) is considered a comment.
///
/// Registers:
/// OWO
/// UWU
/// ONO
/// UNU
/// MEW (Meta Register)
/// Values assigned to this meta-register will be printed to the console as an ASCII character.
/// Using this meta-register as an instruction argument will prompt that an optional number be requested from the console.
/// DMW (Meta Register)
/// Values assigned to this meta-register will be printed to the console.
/// Using this meta-register as an instruction argument will prompt that a required number be requested from the console.
/// BEP
/// BEP is a pointer register that points to the current instruction.
/// BAP (META REGISTER)
/// BAP it refers to the current stack position. you can access to the stack with this register.
/// Instructions:
/// pet:
/// arguments: 2 (register, value)
/// description: assigns a value to a register
/// example: pet OWO 1
/// example: pet UWU 2
/// paw:
/// arguments: 2 (register, value)
/// description: adds a value to a register
/// example: paw OWO 1
/// example: paw UWU 2
/// bop:
/// arguments: 2 (register, value)
/// description: subtracts a value from a register
/// example: bop OWO 1
/// example: bop UWU 2
/// lik:
/// arguments: 2 (register, value)
/// description: multiplies a register by a value
/// example: lik OWO 1
/// example: lik UWU 2
/// kis:
/// arguments: 2 (register, value)
/// description: divides a register by a value
/// example: kis OWO 1
/// example: kis UWU 2
/// bte:
/// arguments: 2 (register, value)
/// description: modulus a register by a value
/// example: bte OWO 1
/// example: bte UWU 2
/// cyt:
/// arguments: 3 (register,value,value)
/// description: if value 1 is greater than value 2, set register to 0
/// example: cyt OWO 1 2
/// example: cyt UWU 2 1
/// example: cyt ONO 1 1
/// example: cyt UNU 1 1
/// wag;
/// arguments: 3 (register,value,value)
/// description: if value 1 is equal value 2, set register to 0
/// example: wag OWO 1 2
/// example: wag UWU 2 1
/// example: wag ONO 1 1
/// example: wag UNU 1 1
/// pnc:
/// arguments: 1 (pointer)
/// description: pushes pointer to the next instruction to the stack and jumps to the pointer
/// example: pnc 1
/// example: pnc @name
/// wig:
/// arguments: 1 (pointer)
/// description: jumps to the pointer
/// example: wig 1
/// example: wig @name
/// nuz:
/// arguments: 0
/// description: pops a pointer from the stack and jumps to it
/// pat:
/// arguments: 1 (register)
/// description: Skips the next instruction if the register is 0
/// example: pat OWO
/// example: pat UWU
/// example: pat ONO
/// yif:
/// arguments: 0
/// description: terminates the program
/// example: yif
/// nom:
/// arguments: 2 (register, value)
/// description: allocates a value in the heap and stores the pointer in the register (the value is the size of the allocation, the allocation is [value]u32)
/// example: nom OWO 1
/// example: nom UWU 2
/// ror:
/// arguments: 2 (register, value)
/// description: reads a value from the heap and stores it in the register (The register has the handle to the heap)
/// example: ror OWO 1
/// example: ror UWU 2
/// yit:
/// arguments: 2 (registerValue)
/// description: frees a value from the heap (The register has the handle to the heap value)
/// example: yit OWO
///
/// Macros:
/// print:
/// arguments: 0
/// description: prints the value of the meta-register MEW in a series of pet instructions
/// example: @print = string values in here
/// values can be registers or int literals
/// pointers can be int literals or labels
/// labels can be any string that does not contain whitespace followed by a colon (:)
/// Registers are insensitive to case
/// Labels are insensitive to case
const FurAsm = struct {
    const Tokens = enum {
        invalid,
        register,
        literal,
        label,
        instruction,
        macro,
        comment,
        newline,
        macro_value,
        whitespace,
        identifier,
        string,
    };

    const Instruction = enum {
        pet,
        paw,
        bop,
        lik,
        kis,
        bte,
        cyt,
        wag,
        pnc,
        nuz,
        wig,
        pat,
        yif,
        yit,
        nom,
        ror,
    };

    const Register = enum {
        owo,
        uwu,
        ono,
        unu,
        mew,
        dmw,
        bep,
        bap,
    };
    const Pattern = ptk.Pattern(Tokens);
    const Tokenizer = ptk.Tokenizer(Tokens, &[_]Pattern{
        Pattern.create(.instruction, ptk.matchers.literal("pet")),
        Pattern.create(.instruction, ptk.matchers.literal("paw")),
        Pattern.create(.instruction, ptk.matchers.literal("bop")),
        Pattern.create(.instruction, ptk.matchers.literal("lik")),
        Pattern.create(.instruction, ptk.matchers.literal("kis")),
        Pattern.create(.instruction, ptk.matchers.literal("bte")),
        Pattern.create(.instruction, ptk.matchers.literal("cyt")),
        Pattern.create(.instruction, ptk.matchers.literal("wag")),
        Pattern.create(.instruction, ptk.matchers.literal("pnc")),
        Pattern.create(.instruction, ptk.matchers.literal("pat")),
        Pattern.create(.instruction, ptk.matchers.literal("nuz")),
        Pattern.create(.instruction, ptk.matchers.literal("wig")),
        Pattern.create(.instruction, ptk.matchers.literal("yif")),
        Pattern.create(.instruction, ptk.matchers.literal("nom")),
        Pattern.create(.instruction, ptk.matchers.literal("ror")),
        Pattern.create(.instruction, ptk.matchers.literal("yit")),
        Pattern.create(.register, ptk.matchers.literal("owo")),
        Pattern.create(.register, ptk.matchers.literal("uwu")),
        Pattern.create(.register, ptk.matchers.literal("ono")),
        Pattern.create(.register, ptk.matchers.literal("unu")),
        Pattern.create(.register, ptk.matchers.literal("mew")),
        Pattern.create(.register, ptk.matchers.literal("dmw")),
        Pattern.create(.register, ptk.matchers.literal("bep")),
        Pattern.create(.register, ptk.matchers.literal("bap")),
        Pattern.create(.literal, ptk.matchers.decimalNumber),
        Pattern.create(.literal, ptk.matchers.binaryNumber),
        Pattern.create(.literal, ptk.matchers.octalNumber),
        Pattern.create(.literal, ptk.matchers.hexadecimalNumber),
        Pattern.create(.macro, ptk.matchers.sequenceOf(.{ ptk.matchers.literal("@"), ptk.matchers.identifier })),
        Pattern.create(.newline, ptk.matchers.linefeed),
        Pattern.create(.macro_value, ptk.matchers.literal("=")),
        Pattern.create(.whitespace, ptk.matchers.whitespace),
        Pattern.create(.comment, ptk.matchers.sequenceOf(.{ ptk.matchers.literal(";"), anyUntilLinefeed })),
        Pattern.create(.label, ptk.matchers.sequenceOf(.{ ptk.matchers.identifier, ptk.matchers.literal(":") })),
        Pattern.create(.identifier, ptk.matchers.sequenceOf(.{ptk.matchers.identifier})),
    });

    const Parser = ptk.ParserCore(Tokenizer, .{ .whitespace, .comment, .newline });
    const ruleSet = ptk.RuleSet(Tokens);

    parser: Parser,
    context: VirtualMachine,
    allocated: bool = false,
    identifier: std.StringArrayHashMap(u32),

    pub fn init(allocator: std.mem.Allocator, program_source_path: []const u8) !FurAsm {
        var context = try VirtualMachine.init(allocator);
        var program_source = try std.fs.cwd().openFile(program_source_path, .{ .read = true });
        defer program_source.deinit();
        var program_source_contents = try program_source.readToEndAlloc(allocator, std.math.maxInt(usize));
        var tokens = try allocator.create(Tokenizer);
        tokens.* = Tokenizer.init(program_source_contents, program_source_path);
        var return_val = FurAsm{
            .parser = Parser.init(tokens),
            .context = context,
            .allocated = true,
            .identifier = std.StringArrayHashMap(u32).init(allocator),
        };

        return_val.parser = Parser.init(tokens);
        return return_val;
    }

    pub fn initBuffer(allocator: std.mem.Allocator, program_source: []const u8) !FurAsm {
        var context = try VirtualMachine.init(allocator);
        var tokens = try allocator.create(Tokenizer);
        tokens.* = Tokenizer.init(program_source, "buffer");
        var return_val = FurAsm{
            .parser = Parser.init(tokens),
            .context = context,
            .allocated = true,
            .identifier = std.StringArrayHashMap(u32).init(allocator),
        };
        return return_val;
    }

    pub fn parse(self: *FurAsm) ![]Instructions {
        var arena = std.heap.ArenaAllocator.init(self.context.allocator);
        var instructions = std.ArrayList(Instructions).init(arena.allocator());
        defer instructions.deinit();
        var current_instruction: Instructions = undefined;
        while (try self.parser.nextToken()) |tok| {
            var token: Tokenizer.Token = undefined;
            if (tok.type == .macro) {
                var state = self.parser.saveState();
                errdefer self.parser.restoreState(state);
                var macro_name_slice = tok.text;
                _ = try self.parser.accept(comptime ruleSet.is(.macro_value));
                var value_start = self.parser.tokenizer.offset;
                var search_newline = anyUntilLinefeed(self.parser.tokenizer.source[value_start..]) orelse return error.MacroValueNoNewline;
                self.parser.tokenizer.offset += search_newline;
                var value_end = self.parser.tokenizer.offset;
                var value_length = value_end - value_start;
                var value = self.parser.tokenizer.source[value_start..][0..value_length];
                current_instruction = Instructions{
                    .macro = .{
                        .name = macro_name_slice[1..],
                        .value = value,
                    },
                };
            } else if (tok.type == .label) {
                var state = self.parser.saveState();
                errdefer self.parser.restoreState(state);
                var label = std.mem.trim(u8, tok.text, ":");
                try self.identifier.put(label, @as(u32, @truncate(instructions.items.len - 1)));
                continue;
            } else if (tok.type == .instruction) {
                var state = self.parser.saveState();
                errdefer self.parser.restoreState(state);
                var instruction = tok.text;
                var instruction_enum = std.meta.stringToEnum(Instruction, instruction) orelse unreachable;
                switch (instruction_enum) {
                    .pet => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));
                        current_instruction = Instructions{
                            .pet = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .paw => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));
                        current_instruction = Instructions{
                            .paw = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .bop => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));
                        current_instruction = Instructions{
                            .bop = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .lik => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));

                        current_instruction = Instructions{
                            .lik = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .kis => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));

                        current_instruction = Instructions{
                            .kis = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .bte => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));

                        current_instruction = Instructions{
                            .bte = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .cyt => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value1_str = token.text;
                        var value1 = try parseValueOrRegister(std.mem.trim(u8, value1_str, " \t"));
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value2_str = token.text;
                        var value2 = try parseValueOrRegister(std.mem.trim(u8, value2_str, " \t"));

                        current_instruction = Instructions{
                            .cyt = .{
                                .register = register,
                                .value1 = value1,
                                .value2 = value2,
                            },
                        };
                    },
                    .wag => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value1_str = token.text;
                        var value1 = try parseValueOrRegister(std.mem.trim(u8, value1_str, " \t"));
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value2_str = token.text;
                        var value2 = try parseValueOrRegister(std.mem.trim(u8, value2_str, " \t"));
                        current_instruction = Instructions{
                            .wag = .{
                                .register = register,
                                .value1 = value1,
                                .value2 = value2,
                            },
                        };
                    },
                    .pnc => {
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .identifier }));
                        var pointer_str = token.text;
                        var pointer_int = parseValueOrRegister(std.mem.trim(u8, pointer_str, " \t")) catch null;
                        if (pointer_int == null) {
                            var pointer_val = PointerValue{
                                .pointer = pointer_str,
                            };
                            current_instruction = Instructions{
                                .pnc = .{
                                    .pointer = pointer_val,
                                },
                            };
                        } else {
                            current_instruction = Instructions{
                                .pnc = .{
                                    .pointer = PointerValue{
                                        .value = pointer_int.?.value,
                                    },
                                },
                            };
                        }
                    },
                    .wig => {
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .identifier }));
                        var pointer_str = token.text;
                        var pointer_int = parseValueOrRegister(std.mem.trim(u8, pointer_str, " \t")) catch null;
                        if (pointer_int == null) {
                            var pointer_val = PointerValue{
                                .pointer = pointer_str,
                            };
                            current_instruction = Instructions{
                                .wig = .{
                                    .pointer = pointer_val,
                                },
                            };
                        } else {
                            current_instruction = Instructions{
                                .wig = .{
                                    .pointer = PointerValue{
                                        .value = pointer_int.?.value,
                                    },
                                },
                            };
                        }
                    },

                    .nuz => {
                        current_instruction = .nuz;
                    },
                    .yif => {
                        current_instruction = .yif;
                    },
                    .pat => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        current_instruction = Instructions{
                            .pat = .{
                                .register = register,
                            },
                        };
                    },
                    .nom => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));
                        current_instruction = Instructions{
                            .nom = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .ror => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        token = try self.parser.accept(comptime ruleSet.oneOf(.{ .literal, .register }));
                        var value_str = token.text;
                        var value = try parseValueOrRegister(std.mem.trim(u8, value_str, " \t"));
                        current_instruction = Instructions{
                            .ror = .{
                                .register = register,
                                .value = value,
                            },
                        };
                    },
                    .yit => {
                        token = try self.parser.accept(comptime ruleSet.is(.register));
                        var register = getRegisterFromText(token.text);
                        current_instruction = Instructions{
                            .yit = .{
                                .register = register,
                            },
                        };
                    },
                }
            }
            try instructions.append(current_instruction);
        }
        return self.context.allocator.dupe(Instructions, instructions.items);
    }

    pub fn run(self: *FurAsm) !void {
        try self.context.setProgram(try self.parse());
        self.context.map = self.identifier;
        try self.context.run();
    }
    fn getRegisterFromText(text: []const u8) Register {
        return std.meta.stringToEnum(Register, text) orelse unreachable;
    }

    fn parseValueOrRegister(text: []const u8) !RegisterValue {
        if (text.len != 3) {
            return RegisterValue{ .value = try parseInt(text) };
        }
        // We check is not a number first because it is more likely to be a register
        if (parseInt(text) catch null) |val| {
            return RegisterValue{
                .value = val,
            };
        } else {
            return RegisterValue{
                .register = getRegisterFromText(text),
            };
        }
    }

    pub fn deinit(self: *FurAsm) void {
        self.context.deinit();
        if (self.allocated) {
            self.context.allocator.free(self.context.program);
        }
    }
};

const RegisterValue = union(enum) {
    register: FurAsm.Register,
    value: u32,
};
const PointerValue = union(enum) {
    pointer: []const u8,
    value: u32,
};

const Instructions = union(enum) {
    pet: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    paw: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    bop: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    lik: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    kis: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    bte: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    cyt: struct {
        register: FurAsm.Register,
        value1: RegisterValue,
        value2: RegisterValue,
    },
    wag: struct {
        register: FurAsm.Register,
        value1: RegisterValue,
        value2: RegisterValue,
    },
    pnc: struct {
        pointer: PointerValue,
    },
    nuz: void,
    wig: struct {
        pointer: PointerValue,
    },
    pat: struct {
        register: FurAsm.Register,
    },
    yif: void,
    nom: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    ror: struct {
        register: FurAsm.Register,
        value: RegisterValue,
    },
    yit: struct {
        register: FurAsm.Register,
    },
    macro: struct {
        name: []const u8,
        value: []const u8,
    },
};
const BufferOut = std.io.BufferedWriter(128 * 1024, std.fs.File.Writer);
const VirtualMachine = struct {
    registers: [4]u32,
    meta_registers: [2]u32,
    instruction_pointer: u32 = 0,
    program: []Instructions,
    allocator: std.mem.Allocator,
    stack: std.ArrayListUnmanaged(u32) = .{},
    heap: std.ArrayListUnmanaged([]u32) = .{},
    stdout: std.fs.File,
    stdin: std.fs.File,
    stdout_buffered: BufferOut,
    map: std.StringArrayHashMap(u32),
    pub fn initProgram(allocator: std.mem.Allocator, program: []Instructions) !VirtualMachine {
        var buffered_stdout = BufferOut{ .unbuffered_writer = std.io.getStdOut().writer() };
        var vm = VirtualMachine{
            .registers = undefined,
            .meta_registers = undefined,
            .map = undefined,
            .instruction_pointer = 0,
            .program = program,
            .allocator = allocator,
            .stdout = std.io.getStdOut(),
            .stdin = std.io.getStdIn(),
            .stdout_buffered = buffered_stdout,
        };

        @call(.always_inline, VirtualMachine.setProgram, .{ &vm, program });
        return vm;
    }

    pub fn init(allocator: std.mem.Allocator) !VirtualMachine {
        var buffered_stdout = BufferOut{ .unbuffered_writer = std.io.getStdOut().writer() };
        return VirtualMachine{
            .registers = undefined,
            .meta_registers = undefined,
            .map = undefined,
            .instruction_pointer = 0,
            .program = undefined,
            .allocator = allocator,
            .stdout_buffered = buffered_stdout,
            .stdout = std.io.getStdOut(),
            .stdin = std.io.getStdIn(),
        };
    }

    pub fn setProgram(self: *VirtualMachine, program: []Instructions) !void {
        self.program = program;
        const max_memory: usize = mem: {
            var max_stack_size: usize = 0;
            for (program) |instruction| {
                switch (instruction) {
                    .pnc => max_stack_size += 1,
                    else => {},
                }
            }
            break :mem max_stack_size;
        };
        if (self.stack.capacity != 0) {
            self.stack.deinit(self.allocator);
        }
        if (self.stack.capacity < max_memory) {
            self.stack = try std.ArrayListUnmanaged(u32).initCapacity(self.allocator, max_memory);
        }
    }

    pub inline fn setRegister(self: *VirtualMachine, register: FurAsm.Register, val: u32) !void {
        return switch (register) {
            inline .owo => self.registers[0] = val,
            inline .uwu => self.registers[1] = val,
            inline .ono => self.registers[2] = val,
            inline .unu => self.registers[3] = val,
            inline .mew => {
                var char: [1]u8 = undefined;
                char[0] = @as(u8, @truncate(val));
                try self.stdout_buffered.writer().writeAll(&char);
            },
            inline .dmw => {
                var chars: [std.math.log(u32, 10, std.math.maxInt(u32)) + 1]u8 = undefined;
                var value = val;
                var len = std.math.log(u32, 10, val) + 1;
                while (value != 0) : (value /= 10) {
                    len -= 1;
                    chars[len] = @as(u8, @truncate(value % 10)) + '0';
                }
                try self.stdout_buffered.writer().writeAll(chars[0 .. std.math.log(u32, 10, val) + 1]);
            },
            inline .bep => self.instruction_pointer = val,
            inline .bap => try self.stack.append(self.allocator, val),
        };
    }

    pub inline fn getRegister(self: *VirtualMachine, register: FurAsm.Register) !u32 {
        return switch (register) {
            inline .owo => self.registers[0],
            inline .uwu => self.registers[1],
            inline .ono => self.registers[2],
            inline .unu => self.registers[3],
            inline .mew => {
                var value: u32 = 0;
                var read_buffer: [std.math.log(u32, 10, std.math.maxInt(u32)) + 1]u8 = undefined;
                var read_bytes: usize = 0;
                while (true) {
                    var byte: [1]u8 = undefined;
                    var read_byte = self.stdin.read(&byte) catch 0;
                    if (read_byte == 0) {
                        break;
                    }
                    if (byte[0] == '\n') {
                        break;
                    }
                    if (read_bytes >= read_buffer.len) {
                        break;
                    }
                    read_buffer[read_bytes] = byte[0];
                    read_bytes += 1;
                }
                value = try parseInt(read_buffer[0..read_bytes]);
                return value;
            },
            inline .dmw => {
                var value: u32 = 0;
                var read_buffer: [std.math.log(u32, 10, std.math.maxInt(u32)) + 1]u8 = undefined;
                var read_bytes: usize = 0;
                while (true) {
                    var byte: [1]u8 = undefined;
                    var read_byte = self.stdin.read(&byte) catch 0;
                    if (read_byte == 0) {
                        break;
                    }
                    if (byte[0] == '\n') {
                        break;
                    }
                    if (read_bytes >= read_buffer.len) {
                        break;
                    }
                    read_buffer[read_bytes] = byte[0];
                    read_bytes += 1;
                }
                value = try parseInt(read_buffer[0..read_bytes]);
                return value;
            },
            inline .bep => self.instruction_pointer,
            inline .bap => self.stack.pop(),
        };
    }
    const MathOp = enum { add, sub, mul, div, mod };

    inline fn doMath(value1: u32, value2: u32, comptime operation: MathOp) u32 {
        return switch (operation) {
            inline .add => value1 +% value2,
            inline .sub => value1 -% value2,
            inline .mul => value1 *% value2,
            inline .div => @divTrunc(value1, value2),
            inline .mod => @mod(value1, value2),
        };
    }

    inline fn getValue(self: *VirtualMachine, value: RegisterValue) !u32 {
        return switch (value) {
            inline .register => try self.getRegister(value.register),
            inline .value => value.value,
        };
    }

    inline fn getPointer(self: *VirtualMachine, pointer: PointerValue) !u32 {
        return switch (pointer) {
            inline .pointer => self.map.get(pointer.pointer) orelse error.MissingLabel,
            inline .value => pointer.value,
        };
    }

    pub fn run(self: *VirtualMachine) !void {
        while (self.instruction_pointer < self.program.len) : (self.instruction_pointer += 1) {
            const instruction = self.program[self.instruction_pointer];
            switch (instruction) {
                inline .yif => {
                    try self.stdout_buffered.flush();
                    return;
                },
                .pet => |pet_instruction| {
                    const register = pet_instruction.register;
                    const value = try self.getValue(pet_instruction.value);
                    try self.setRegister(register, value);
                },
                .paw => |paw_instruction| {
                    const register = paw_instruction.register;
                    const value = try self.getValue(paw_instruction.value);
                    const current_value = try self.getRegister(register);
                    const new_value = VirtualMachine.doMath(current_value, value, .add);
                    try self.setRegister(register, new_value);
                },
                .bop => |bop_instruction| {
                    const register = bop_instruction.register;
                    const value = try self.getValue(bop_instruction.value);
                    const current_value = try self.getRegister(register);
                    const new_value = VirtualMachine.doMath(current_value, value, .sub);
                    try self.setRegister(register, new_value);
                },
                .lik => |lik_instruction| {
                    const register = lik_instruction.register;
                    const value = try self.getValue(lik_instruction.value);
                    const current_value = try self.getRegister(register);
                    const new_value = VirtualMachine.doMath(current_value, value, .mul);
                    try self.setRegister(register, new_value);
                },
                .kis => |kis_instruction| {
                    const register = kis_instruction.register;
                    const value = try self.getValue(kis_instruction.value);
                    const current_value = try self.getRegister(register);
                    const new_value = VirtualMachine.doMath(current_value, value, .div);
                    try self.setRegister(register, new_value);
                },
                .bte => |bte_instruction| {
                    const register = bte_instruction.register;
                    const value = try self.getValue(bte_instruction.value);
                    const current_value = try self.getRegister(register);
                    const new_value = VirtualMachine.doMath(current_value, value, .mod);
                    try self.setRegister(register, new_value);
                },
                .cyt => |cyt_instruction| {
                    const register = cyt_instruction.register;
                    const value1 = try self.getValue(cyt_instruction.value1);
                    const value2 = try self.getValue(cyt_instruction.value2);
                    const current_value = try self.getRegister(register);
                    const new_value = if (value1 > value2) 0 else current_value;
                    try self.setRegister(register, new_value);
                },
                .wag => |wag_instruction| {
                    const register = wag_instruction.register;
                    const value1 = try self.getValue(wag_instruction.value1);
                    const value2 = try self.getValue(wag_instruction.value2);

                    const current_value = try self.getRegister(register);
                    const new_value = if (value1 == value2) 0 else current_value;
                    try self.setRegister(register, new_value);
                },
                .pnc => |pnc_instruction| {
                    const pointer = try self.getPointer(pnc_instruction.pointer);
                    try self.stack.append(self.allocator, self.instruction_pointer);
                    self.instruction_pointer = pointer -| 1;
                },
                .wig => |wig_instruction| {
                    const pointer = try self.getPointer(wig_instruction.pointer);
                    self.instruction_pointer = pointer;
                },
                .pat => |pat_instruction| {
                    const register = pat_instruction.register;
                    const value = try self.getRegister(register);
                    if (value == 0) {
                        self.instruction_pointer += 1;
                    }
                },
                .nuz => {
                    const pointer = self.stack.pop();
                    self.instruction_pointer = pointer;
                },
                .macro => {
                    if (std.mem.eql(u8, instruction.macro.name, "print")) {
                        try self.stdout_buffered.writer().writeAll(std.mem.trim(u8, instruction.macro.value, " \t"));
                    }
                },
                .nom => |nom_instruction| {
                    const register = nom_instruction.register;
                    var value = try self.getValue(nom_instruction.value);
                    // We allocate N and save the index to the register
                    try self.heap.append(self.allocator, try self.allocator.alloc(u32, value));
                    try self.setRegister(register, @as(u32, @truncate(self.heap.items[self.heap.items.len - 1].len)));
                },
                .ror => |ror_instruction| {
                    const register = ror_instruction.register;
                    var value = try self.getValue(ror_instruction.value);
                    var current_value = try self.getRegister(register);
                    var current_heap_value = self.heap.items[current_value];

                    try self.setRegister(register, current_heap_value[value]);
                },
                .yit => |yit_instruction| {
                    const register = yit_instruction.register;
                    var current_value = try self.getRegister(register);
                    var current_heap_value = self.heap.items[current_value];
                    self.allocator.free(current_heap_value);
                },
            }
        }
        return error.not_properly_terminated;
    }

    pub fn deinit(self: *VirtualMachine) void {
        self.stack.deinit(self.allocator);
    }
};

fn parseInt(buffer: []const u8) !u32 {
    var value: u32 = 0;
    for (buffer) |byte| {
        if (std.ascii.isDigit(byte)) {
            value *= 10;
            value += byte - '0';
        } else {
            return error.InvalidNumber;
        }
    }
    return value;
}

fn anyUntilLinefeed(characters: []const u8) ?usize {
    var i: usize = 0;
    while (i < characters.len) : (i += 1) {
        if (characters[i] == '\n') {
            return i;
        }
        if (characters[i] == '\r' and i + 1 < characters.len and characters[i + 1] == '\n') {
            return i + 1;
        }
    }
    return null;
}

/// This is another language that will be on top of FurAsm
/// It will be a more high level language that will compile to FurAsm
/// Then FurAsm will execute it via the VirtualMachine
/// The language is called Furcode
/// TODO: WRITE THIS SOME DAY
const FurCode = struct {};
