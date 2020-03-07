#include "pch.h"
#include "Camera.h"
#include "Component.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "MeshPlane.h"

enum class SerializationEnum1
{
    Value1 = 0,
    Value2 = 1,
    Value3,
    Value4
};

class SerializationClass1
{
    public:
    float field1{};
    float field2{};
    SerializationEnum1 field3{SerializationEnum1::Value2};
    SerializationEnum1 field4{SerializationEnum1::Value3};
    glm::vec2 vec2{};
    glm::vec3 vec3{};
    glm::vec4 vec4{};
    glm::mat2 mat2{};
    glm::mat3 mat3{};
    glm::mat4 mat4{};
    std::vector<int> intVector{};
    RTTR_ENABLE();
};

class SerializationClass2
{
    public:
    SerializationClass2() = default;
    SerializationClass2(const std::shared_ptr<SerializationClass1>& class1) : shared(class1), weak(class1), raw(class1.get()) {}
    std::shared_ptr<SerializationClass1> getWeak() const
    {
        return weak.lock();
    }
    void setWeak(std::shared_ptr<SerializationClass1> shared)
    {
        weak = shared;
    }
    int field1{};
    int field2{};
    std::shared_ptr<SerializationClass1> shared{};
    std::weak_ptr<SerializationClass1> weak{};
    SerializationClass1* raw{nullptr};
    std::vector<std::shared_ptr<SerializationClass1>> ptrVector{};
    std::map<int, int> intMap{};
    RTTR_ENABLE();
};

class SerializationComponent1 : public spark::Component
{
    public:
    SerializationComponent1() = default;
    void update() override{};
    void fixedUpdate() override{};
    void drawGUI() override{};
    RTTR_ENABLE_NULL_COMPONENT(SerializationComponent1);
    RTTR_ENABLE(Component);
};

class SerializationComponent2 : public spark::Component
{
    public:
    SerializationComponent2() = default;
    void update() override{};
    void fixedUpdate() override{};
    void drawGUI() override{};
    std::shared_ptr<SerializationComponent1> shared{std::make_shared<SerializationComponent1>()};
    std::shared_ptr<Component> sharedComp{shared};
    RTTR_ENABLE(Component);
};

struct SerializationStruct1
{
    glm::mat3 mat;
    double d;
    glm::vec4 vec;
    RTTR_ENABLE();
};

struct SerializationStruct2
{
    int i;
    float f;
    glm::ivec2 ivec;
    SerializationStruct1 s;
    RTTR_ENABLE();
};

RTTR_REGISTRATION
{
    rttr::registration::enumeration<SerializationEnum1>("SerializationEnum1")(
        rttr::value("Value1", SerializationEnum1::Value1), rttr::value("Value2", SerializationEnum1::Value2),
        rttr::value("Value3", SerializationEnum1::Value3), rttr::value("Value4", SerializationEnum1::Value4));

    rttr::registration::class_<SerializationClass1>("SerializationClass1")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("field1", &SerializationClass1::field1)
        .property("field2", &SerializationClass1::field2)
        .property("field3", &SerializationClass1::field3)
        .property("field4", &SerializationClass1::field4)
        .property("vec2", &SerializationClass1::vec2)
        .property("vec3", &SerializationClass1::vec3)
        .property("vec4", &SerializationClass1::vec4)
        .property("mat2", &SerializationClass1::mat2)
        .property("mat3", &SerializationClass1::mat3)
        .property("mat4", &SerializationClass1::mat4)
        .property("intVector", &SerializationClass1::intVector);

    rttr::registration::class_<SerializationClass2>("SerializationClass2")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("field1", &SerializationClass2::field1)
        .property("field2", &SerializationClass2::field2)
        .property("shared", &SerializationClass2::shared)
        .property("weak", &SerializationClass2::getWeak, &SerializationClass2::setWeak, rttr::registration::public_access)
        .property("raw", &SerializationClass2::raw)
        .property("ptrVector", &SerializationClass2::ptrVector)
        .property("intMap", &SerializationClass2::intMap);

    rttr::registration::class_<SerializationComponent1>("SerializationComponent1")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr) RTTR_REGISTER_NULL_COMPONENT(SerializationComponent1);

    rttr::registration::class_<SerializationComponent2>("SerializationComponent2")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("shared", &SerializationComponent2::shared)
        .property("sharedComp", &SerializationComponent2::sharedComp);

    rttr::registration::class_<SerializationStruct1>("SerializationStruct1")
        .constructor()(rttr::policy::ctor::as_object)
        .property("mat", &SerializationStruct1::mat)
        .property("d", &SerializationStruct1::d)
        .property("vec", &SerializationStruct1::vec);

    rttr::registration::class_<SerializationStruct2>("SerializationStruct2")
        .constructor()(rttr::policy::ctor::as_object)
        .property("i", &SerializationStruct2::i)
        .property("f", &SerializationStruct2::f)
        .property("ivec", &SerializationStruct2::ivec)
        .property("s", &SerializationStruct2::s);
}

TEST(SerializationTest, PointersInterchangeable)
{
    std::shared_ptr<SerializationClass1> shared = std::make_shared<SerializationClass1>();
    SerializationClass1* raw = shared.get();
    rttr::variant sharedVar = shared, rawVar = raw, wrappedVar = sharedVar.extract_wrapped_value();

    ASSERT_TRUE(sharedVar.get_type().is_wrapper());
    ASSERT_FALSE(rawVar.get_type().is_wrapper());

    ASSERT_FALSE(sharedVar.get_type().is_pointer());
    ASSERT_TRUE(wrappedVar.get_type().is_pointer());
    ASSERT_TRUE(rawVar.get_type().is_pointer());

    ASSERT_TRUE(wrappedVar.get_type() == rawVar.get_type());

    ASSERT_EQ(static_cast<void*>(raw), rawVar.get_value<void*>());
    ASSERT_EQ(rawVar.get_value<void*>(), wrappedVar.get_value<void*>());
}

TEST(SerializationTest, PointersSerializedProperly)
{
    std::shared_ptr<SerializationClass1> class1 = std::make_shared<SerializationClass1>();
    class1->field1 = 3.0f;
    class1->field2 = 4.0f;
    class1->field3 = SerializationEnum1::Value1;
    class1->field4 = SerializationEnum1::Value4;
    class1->vec2 = {1.0f, 2.0f};
    class1->vec3 = {3.0f, 4.0f, 5.0f};
    class1->vec4 = {6.0f, 7.0f, 8.0f, 9.0f};
    class1->mat2 = {{10.0f, 11.0f}, {12.0f, 13.0f}};
    class1->mat3 = {{14.0f, 15.0f, 16.0f}, {17.0f, 18.0f, 19.0f}, {20.0f, 21.0f, 22.0f}};
    class1->mat4 = {{23.0f, 24.0f, 25.0f, 26.0f}, {27.0f, 28.0f, 29.0f, 30.0f}, {31.0f, 32.0f, 33.0f, 34.0f}, {35.0f, 36.0f, 37.0f, 38.0f}};
    class1->intVector = {4, 4, 6, 6, 8, 8};
    std::shared_ptr<SerializationClass2> class2 = std::make_shared<SerializationClass2>(class1);
    class2->field1 = 100;
    class2->field2 = 500;
    class2->intMap = {{4, 7}, {8, 12}, {99, 100}};
    class2->ptrVector = {class1, class1, class1};
    Json::Value root;
    spark::JsonSerializer* serializer{spark::JsonSerializer::getInstance()};
    ASSERT_TRUE(serializer->save(class2, root));
    spark::JsonSerializer::writeToFile("test.json", root);
    rttr::variant var{serializer->loadVariant(root)};
    ASSERT_TRUE(var.is_valid());
    ASSERT_TRUE(var.is_type<std::shared_ptr<SerializationClass2>>());
    std::shared_ptr<SerializationClass2> deserializedClass2{var.get_value<std::shared_ptr<SerializationClass2>>()};
    ASSERT_EQ(class2->field1, deserializedClass2->field1);
    ASSERT_EQ(class2->field2, deserializedClass2->field2);
    ASSERT_NE(deserializedClass2->raw, nullptr);
    ASSERT_EQ(deserializedClass2->shared.get(), deserializedClass2->raw);
    ASSERT_EQ(deserializedClass2->weak.lock(), deserializedClass2->shared);
    ASSERT_EQ(class2->intMap.size(), deserializedClass2->intMap.size());
    for(auto& item : class2->intMap)
    {
        ASSERT_EQ(item.second, deserializedClass2->intMap[item.first]);
    }
    ASSERT_EQ(class2->ptrVector.size(), deserializedClass2->ptrVector.size());
    for(int i = 0; i < class2->ptrVector.size(); ++i)
    {
        ASSERT_EQ(deserializedClass2->ptrVector[i], deserializedClass2->shared);
    }
    ASSERT_EQ(class2->raw->intVector.size(), deserializedClass2->raw->intVector.size());
    for(int i = 0; i < class2->raw->intVector.size(); ++i)
    {
        ASSERT_EQ(class2->raw->intVector[i], deserializedClass2->raw->intVector[i]);
    }
    ASSERT_EQ(class2->raw->field1, deserializedClass2->raw->field1);
    ASSERT_EQ(class2->raw->field2, deserializedClass2->raw->field2);
    ASSERT_EQ(class2->raw->field3, deserializedClass2->raw->field3);
    ASSERT_EQ(class2->raw->field4, deserializedClass2->raw->field4);
    ASSERT_EQ(class2->raw->vec2, deserializedClass2->raw->vec2);
    ASSERT_EQ(class2->raw->vec3, deserializedClass2->raw->vec3);
    ASSERT_EQ(class2->raw->vec4, deserializedClass2->raw->vec4);
    ASSERT_EQ(class2->raw->mat2, deserializedClass2->raw->mat2);
    ASSERT_EQ(class2->raw->mat3, deserializedClass2->raw->mat3);
    ASSERT_EQ(class2->raw->mat4, deserializedClass2->raw->mat4);
}

TEST(SerializationTest, StructsSerializedProperly)
{
    SerializationStruct2 source;
    source.i = 100;
    source.f = 50.0f;
    source.ivec = {-5, 5};
    source.s.d = 0.1234f;
    source.s.vec = {0.1f, 0.2f, 0.3f, 0.4f};
    source.s.mat = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
    spark::JsonSerializer* serializer{spark::JsonSerializer::getInstance()};
    Json::Value root;
    ASSERT_TRUE(serializer->save(source, root));
    spark::JsonSerializer::writeToFile("test2.json", root);
    SerializationStruct2 target;
    try
    {
        target = serializer->loadJson<SerializationStruct2>(root);
    }
    catch(std::exception&)
    {
        ASSERT_FALSE("Unable to deserialize struct!");
    }
    ASSERT_EQ(source.f, target.f);
    ASSERT_EQ(source.i, target.i);
    ASSERT_EQ(source.ivec, target.ivec);
    ASSERT_EQ(source.s.d, target.s.d);
    ASSERT_EQ(source.s.mat, target.s.mat);
    ASSERT_EQ(source.s.vec, target.s.vec);
}

TEST(SerializationTest, ComponentPointersConvertible)
{
    std::shared_ptr<spark::Camera> cam = std::make_shared<spark::Camera>();
    Json::Value root;
    spark::JsonSerializer* serializer{spark::JsonSerializer::getInstance()};
    ASSERT_TRUE(serializer->save(cam, root));
    std::shared_ptr<spark::Component> comp1{cam}, comp2;
    try
    {
        comp2 = serializer->loadJsonShared<spark::Component>(root);
    }
    catch(std::exception&)
    {
        ASSERT_FALSE("Unable to deserialize camera as component!");
    }
    ASSERT_NE(nullptr, comp2);
    ASSERT_EQ(rttr::instance(rttr::variant(cam).extract_wrapped_value()).get_derived_type(),
              rttr::instance(rttr::variant(comp2).extract_wrapped_value()).get_derived_type());
}

TEST(SerializationTest, GameObjectWithComponentSerializedProperly)
{
    std::shared_ptr<spark::GameObject> obj = std::make_shared<spark::GameObject>("TestObj");
    obj->addComponent(std::make_shared<spark::Camera>());
    Json::Value root;
    spark::JsonSerializer* serializer{spark::JsonSerializer::getInstance()};
    ASSERT_TRUE(serializer->save(obj, root));
    spark::JsonSerializer::writeToFile("test3.json", root);
    std::shared_ptr<spark::GameObject> obj2;
    try
    {
        obj2 = serializer->loadJsonShared<spark::GameObject>(root);
    }
    catch(std::exception&)
    {
        ASSERT_FALSE("Unable to deserialize gameobject!");
    }
    ASSERT_EQ(obj->getName(), obj2->getName());
    ASSERT_NE(nullptr, obj->getComponent<spark::Camera>());
    ASSERT_NE(nullptr, obj2->getComponent<spark::Camera>());
}

TEST(SerializationTest, ComponentNullSmartPointersInjectedProperly)
{
    std::shared_ptr<SerializationComponent2> source{std::make_shared<SerializationComponent2>()}, target{};
    ASSERT_NE(nullptr, source->shared.get());
    ASSERT_NE(nullptr, source->sharedComp.get());
    source->shared.reset();
    source->sharedComp.reset();
    ASSERT_EQ(nullptr, source->shared.get());
    ASSERT_EQ(nullptr, source->sharedComp.get());
    Json::Value root;
    spark::JsonSerializer* serializer{spark::JsonSerializer::getInstance()};
    ASSERT_TRUE(serializer->save(source, root));
    spark::JsonSerializer::writeToFile("test4.json", root);
    try
    {
        target = serializer->loadJsonShared<SerializationComponent2>(root);
    }
    catch(std::exception&)
    {
        ASSERT_FALSE("Unable to deserialize component!");
    }
    ASSERT_EQ(nullptr, target->shared.get());
    ASSERT_EQ(nullptr, target->sharedComp.get());
}